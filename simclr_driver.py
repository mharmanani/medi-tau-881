import torch
torch.manual_seed(4832748)

from src.modeling.registry import create_model
from src.typing import FeatureExtractionProtocol
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

CHECKPOINTS_DIR_MODEL = '../../../../simclr/{0}.pt'
CHECKPOINTS_DIR_CLF = '../../../../simclr_clf/{0}.pt'

class SimCLRDriver:
    def __init__(self, device, batch_size, 
                 train_epochs, finetune_epochs,
                 t, eps, out_dim, hidden_dim,
                 use_final_bn, use_dropout, 
                 learning_rate, backbone_name, datamodule,
                 sl_datamodule):
        
        self.device = device
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.finetune_epochs = finetune_epochs
        self.t = t
        self.eps = eps
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim

        self.use_final_bn = use_final_bn
        self.use_dropout = use_dropout
        self.backbone_name = backbone_name

        self.datamodule = datamodule
        self.datamodule.setup()

        self.sl_datamodule = sl_datamodule
        self.sl_datamodule.setup()

        self.learning_rate = learning_rate

    def run(self): 
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
        from torch.utils.data import DataLoader
        from torch import nn
        from torch import optim
        import torch 
        from torch.nn import functional as F
        from tqdm import tqdm
        import wandb

        wandb.init(project="simclr")

        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()

        finetune_loader = self.sl_datamodule.train_dataloader()
        finetune_val_loader = self.sl_datamodule.val_dataloader()
        test_loader = self.sl_datamodule.test_dataloader()

        class SimCLR(nn.Module):
            def __init__(self, device, batch_size,
                 t, eps, out_dim, hidden_dim, learning_rate, use_final_bn, use_dropout, backbone_name):
                super().__init__()

                if backbone_name is None:
                    raise ValueError(f"Must specify either backbone or backbone_name")

                self.device = device

                #from hydra.utils import instantiate

                self.backbone = create_model(backbone_name)
                self.backbone = self.backbone.to(self.device)                

                # implement feature extraction protocol
                self.get_features = nn.Sequential(*list(self.backbone.children())[:-1])
                self.features_dim = self.backbone.fc.in_features

                self.batch_size = batch_size

                self.projector = nn.Sequential(
                    nn.Linear(self.features_dim, self.features_dim),
                    nn.BatchNorm1d(self.features_dim),
                    nn.ReLU(),
                    nn.Linear(self.features_dim, out_dim, bias=False),
                    nn.ReLU()
                )
                self.projector = self.projector.to(self.device)

                self.classifier = nn.Sequential(
                    nn.Linear(self.features_dim, 2),
                    nn.ReLU()
                )
                    
                self.classifier = self.classifier.to(self.device)

                self.clf_loss = nn.CrossEntropyLoss()

                self.linearmodel = LogisticRegression()

                self.t = t
                self.eps = eps
                self.learning_rate = learning_rate

            def forward(self, x):
                h = self.get_features(x)
                h = torch.squeeze(h)
                z = self.projector(h)
                return z

            def NT_Xent(self, z1, z2):
                z1 = F.normalize(z1, p=2, dim=1)
                z2 = F.normalize(z2, p=2, dim=1)
                emb = torch.cat([z1, z2], dim=0)
                n_samples = emb.size(0)
                diag_mask = 1 - torch.eye(n_samples, device=self.device)

                pos_dps = torch.sum(z1*z2, dim=-1) / self.t
                pos_dps = torch.cat([pos_dps, pos_dps], dim=0)


                tril = torch.tril(pos_dps, diagonal=-n_samples//2)
                pos_dps = (pos_dps - tril) * diag_mask

                neg_dps = (emb@emb.T) / self.t
                neg_dps = diag_mask * neg_dps

                #neg_dps = ~pos_dps.bool() * neg_dps

                loss = -pos_dps + torch.logsumexp(neg_dps, dim=-1)
                loss = torch.mean(loss)

                return loss
        
        wandb.log({
            "temp_hyperparam": self.t, 
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": 1e-6
        })
        
        model = SimCLR(device=self.device, 
            t=self.t, eps=self.eps,
            batch_size=self.batch_size, 
            hidden_dim=self.hidden_dim, out_dim=self.out_dim, 
            backbone_name=self.backbone_name,
            use_final_bn=self.use_final_bn, 
            use_dropout=self.use_dropout,
            learning_rate=self.learning_rate
        )

        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs...")
            model.backbone = nn.DataParallel(model.backbone)
            model.classifier = nn.DataParallel(model.classifier)

        wandb.watch(model, log="all")

        optimizer = optim.Adam(model.parameters(), 
                               weight_decay=1e-6,
                               lr=self.learning_rate)
        
        finetuner = optim.Adam(model.classifier.parameters(), 
                              weight_decay=1e-6,
                              lr=self.learning_rate)

        def train(model, device, train_loader, optimizer, epoch):
            model.train()
            losses = []
            for i, batch in enumerate(tqdm(train_loader, desc='Training')):
                X1, X2 = batch[0]
                X1 = X1.to(device)
                X2 = X2.to(device)
                z1 = model(X1)
                z2 = model(X2)
                optimizer.zero_grad()
                loss = model.NT_Xent(z1, z2)
                loss.backward()
                optimizer.step()
                wandb.log({"train_loss": loss})
                losses.append(loss.detach())
            avg_loss = sum(losses) / len(losses)
            wandb.log({"per_epoch_train_loss": avg_loss, "train_epoch": epoch})

        def validate(model, device, val_loader, optimizer, epoch):
            model.eval()
            losses = []
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                X1, X2 = batch[0]
                X1 = X1.to(device)
                X2 = X2.to(device)
                z1 = model(X1)
                z2 = model(X2)
                optimizer.zero_grad()
                loss = model.NT_Xent(z1, z2)
                loss.backward()
                optimizer.step()
                wandb.log({"val_loss": loss})
                losses.append(loss.detach())
            avg_loss = sum(losses) / len(losses)
            wandb.log({"per_epoch_val_loss": avg_loss, "train_epoch": epoch})

        def finetune(model, device, finetune_loader, optimizer, epoch):
            model.eval()
            preds = []
            trues = []
            preds_val = []
            trues_val = []
            for i, batch in enumerate(tqdm(finetune_loader, desc='Finetuning')):
                H = model.get_features(batch[0].to(device))
                y = batch[1].to(device)
                z = model.classifier(H.squeeze())
                finetuner.zero_grad()
                loss = model.clf_loss(z, y.long())
                loss.backward()
                finetuner.step()
                wandb.log({"ft_loss": loss})

                trues += y.detach().to('cpu').numpy().tolist()
                preds += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()

            for k, batch in enumerate(tqdm(finetune_val_loader, desc='Finetuning')):
                H = model.get_features(batch[0].to(device))
                y = batch[1].to(device)
                z = model.classifier(H.squeeze())
                val_loss = model.clf_loss(z, y.long())
                wandb.log({"ft_val_loss_clf_{0}".format(i): loss})

                trues_val += y.detach().to('cpu').numpy().tolist()
                preds_val += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()
                            
            ft_acc = accuracy_score(trues, preds)
            ft_val_acc = accuracy_score(trues_val, preds_val)
            #print("[Epoch {0}] Finetune Accuracy Score: {1}".format(epoch, ft_acc))
            wandb.log({"ft_acc": ft_acc, "ft_epoch": epoch})
            wandb.log({"ft_val_acc": ft_val_acc, "ft_epoch": epoch})

        def make_kNN_OOD_evaluator(model, k, device, train_loader):
            embs = []
            labels = []
            for patch, label, metadata in tqdm(train_loader):
                emb = model.get_features(patch.to(device))
                emb = torch.squeeze(emb)
                embs.append(emb.detach().cpu())
                labels.append(label.detach().cpu())
            
            embs = torch.concat(embs, dim=0).numpy()
            labels = torch.concat(labels, dim=0).numpy()

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(embs, labels)

            return knn

        def test(model, device, test_loader, confidence_t=1.0, ood_eval=None, best_model=None, best_clf=None, log=True):
            model.eval()
            
            if best_model:
                model.load_state_dict(torch.load(CHECKPOINTS_DIR_MODEL.format(best_model)))
            if best_clf:
                model.classifier.load_state_dict(torch.load(CHECKPOINTS_DIR_CLF.format(best_clf)))

            from src.utils.metrics import PatchwiseAndCorewiseMetrics
            metrics_fn = PatchwiseAndCorewiseMetrics(
                compute_corewise_and_patchwise=True, 
                breakdown_by_center=False, 
                metrics=['auroc', 'acc_macro', 'avg_prec', 'brier_score', 'expected_calibration_error']
            )

            from src.utils.metrics import OutputCollector
            output_collector = OutputCollector()

            for patch, label, metadata in tqdm(test_loader):
                with torch.inference_mode():
                    feats = model.get_features(patch.to(device))
                    feats = torch.squeeze(feats)
                    logits = model.classifier(feats)
                    if ood_eval:
                        knn = ood_eval
                        dist, _ = knn.kneighbors(feats.detach().cpu(), return_distance=True)
                        avg_dist = np.mean(dist, axis=1)
                        avg_dist = (avg_dist - avg_dist.min()) / (avg_dist.max() - avg_dist.min())
                        print(avg_dist)
                        includes = torch.Tensor((avg_dist <= confidence_t).astype(int)).squeeze()
                    else:
                        includes = torch.ones(logits.shape[0])
                    indices = torch.nonzero(includes)
                    all_cores = metadata["core_specifier"]
                    pos = metadata["position"]
                    filtered_cores = [all_cores[i] for i in range(len(includes)) if includes[i] != 0]
                output_collector.collect_batch({
                    'logits':logits[indices].squeeze(1), 
                    'labels': label[indices].squeeze(1), 
                    'position': pos[indices].squeeze(1), #**metadata
                    'core_specifier': filtered_cores
                })

            out = output_collector.compute()
            output_collector.reset()

            test_metrics = {}
            metric_fn_out = metrics_fn(out)
            for metric_name in metric_fn_out:
                test_metrics['test/' + metric_name] = metric_fn_out[metric_name].item()
            
            if log:
                wandb.log(test_metrics)

            return test_metrics, len(filtered_cores)
        
        def plot_acc_vs_confidence():
            import pandas as pd
            df = pd.DataFrame()
            for k in [5, 10, 15, 20]:
                knn = make_kNN_OOD_evaluator(model, k=k, device=self.device, train_loader=finetune_loader)
                for kappa in range(0, 11):
                    kappa = kappa / 10
                    metrics, remaining_cores = test(model, self.device, test_loader, ood_eval=knn, confidence_t=kappa, best_model=199, best_clf=99, log=False)
                    log_df = {"confidence_threshold": kappa, "num_neighbours": k, "remaining_cores": remaining_cores}
                    for key in metrics:
                        new_key = key.replace("test/", "")
                        log_df[new_key] = metrics[key]
                    df = df.append(log_df, ignore_index=True)
                
            df.to_csv("../../../../knn_vs_perf_df.csv")

        #model.load_state_dict(torch.load(CHECKPOINTS_DIR_MODEL.format('_simclr_v2_model149')))
        for epoch in range(self.train_epochs):
            train(model, self.device, train_loader, optimizer, epoch)
            validate(model, self.device, val_loader, optimizer, epoch)
            torch.save(model.state_dict(), CHECKPOINTS_DIR_MODEL.format(epoch))
        
        for epoch in range(self.finetune_epochs):
            finetune(model, self.device, finetune_loader, optimizer, epoch)
            torch.save(model.classifier.state_dict(), CHECKPOINTS_DIR_CLF.format(epoch))
        
        #knn = make_kNN_OOD_evaluator(model, k=k, device=self.device, train_loader=finetune_loader)
        test(model, self.device, test_loader, ood_eval=None, confidence_t=1.0, best_model=199, best_clf=99)

        #plot_acc_vs_confidence()
        
        wandb.alert(
            title='Run complete',
            text=f'Your SimCLR model is done training! Check out https://wandb.ai/home for details.',
            level=wandb.AlertLevel.INFO,
        )

    