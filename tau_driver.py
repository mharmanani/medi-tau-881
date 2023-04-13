import torch
torch.manual_seed(88)

from src.modeling.registry import create_model
from src.typing import FeatureExtractionProtocol
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

CHECKPOINTS_DIR_MODEL = '../../../../tau_simclr/{0}.pt'
CHECKPOINTS_DIR_CLF = '../../../../tau_simclr_clf/{0}.pt'

class TaUDriver:
    def __init__(self, device, batch_size, 
                 train_epochs, finetune_epochs,
                 t, eps, out_dim, confidence_threshold,
                 use_final_bn, use_dropout, 
                 learning_rate, weight_decay,
                 backbone_name, datamodule,
                 sl_datamodule):
        
        self.device = device
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.finetune_epochs = finetune_epochs
        self.t = t
        self.weight_decay = weight_decay
        self.eps = eps
        self.out_dim = out_dim
        self.confidence_threshold = confidence_threshold

        self.use_final_bn = use_final_bn
        self.use_dropout = use_dropout
        self.backbone_name = backbone_name

        self.datamodule = datamodule
        self.datamodule.setup()

        self.sl_datamodule = sl_datamodule
        self.sl_datamodule.setup()

        self.learning_rate = learning_rate
        self.tau_stats = {"mean": 3.65, "stdev": 0.9, "max": 0, "min": 0}

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

        wandb.init(project="tau-simclr")

        train_loader = self.datamodule.train_dataloader()
        val_loader = self.datamodule.val_dataloader()

        finetune_loader = self.sl_datamodule.train_dataloader()
        finetune_val_loader = self.sl_datamodule.val_dataloader()
        test_loader = self.sl_datamodule.test_dataloader()

        class TaUSimCLR(nn.Module):
            def __init__(self, device, batch_size, weight_decay,
                 t, eps, out_dim, confidence_threshold, learning_rate,
                 use_final_bn, use_dropout, backbone_name):
                super().__init__()

                if backbone_name is None:
                    raise ValueError(f"Must specify either backbone or backbone_name")

                self.device = device

                #from hydra.utils import instantiate

                self.backbone = create_model(backbone_name)
                self.backbone = self.backbone.to(self.device)
                

                # implement feature extraction protocol
                self.features_dim = self.backbone.fc.in_features
                self.get_features = nn.Sequential(*list(self.backbone.children())[:-1])
                
                self.batch_size = batch_size

                # the 'posterior head' projector for TaU_SimCLR
                self.projector = nn.Sequential(
                    nn.Linear(self.features_dim-1, self.features_dim),
                    nn.BatchNorm1d(self.features_dim),
                    nn.ReLU(),
                    nn.Linear(self.features_dim, out_dim, bias=False), # add 1 for temps
                    nn.ReLU()
                )
                self.projector = self.projector.to(self.device)

                self.classifier = nn.Sequential(
                    nn.Linear(self.features_dim-1, 2).to(self.device),
                    nn.ReLU()
                )
                self.clf_loss = nn.CrossEntropyLoss()

                self.linearmodel = LogisticRegression()

                self.t = t
                self.eps = eps
                self.learning_rate = learning_rate

            def forward(self, x):
                h = self.get_features(x)
                h = torch.squeeze(h)
                h, tau = h[:, :-1], h[:, -1].unsqueeze(1)
                z = self.projector(h)
                return z, tau
            
            def tau_loss(self, z1, z2, temp1, temp2):
                z1 = F.normalize(z1, p=2, dim=1)
                z2 = F.normalize(z2, p=2, dim=1)
                emb = torch.cat([z1, z2], dim=0)
                n_samples = emb.size(0)
                diag_mask = 1 - torch.eye(n_samples, device=self.device)

                scale = self.t
                tau1 = torch.sigmoid(temp1) / scale
                tau2 = torch.sigmoid(temp2) / scale
                tau = torch.cat([tau1, tau2])

                pos_dps = torch.sum(z1*z2, dim=-1) * tau1
                pos_dps = torch.cat([pos_dps, pos_dps], dim=0)
                pos_dps = torch.cat([pos_dps, pos_dps], dim=1)

                tril = torch.tril(pos_dps, diagonal=-n_samples//2)
                pos_dps = (pos_dps - tril) * diag_mask

                neg_dps = (emb@emb.T) * tau
                neg_dps = diag_mask * neg_dps
                #neg_dps = ~pos_dps.bool() * neg_dps

                loss = -pos_dps + torch.logsumexp(neg_dps, dim=-1)
                loss = torch.mean(loss)

                return loss
        
        wandb.log({
            "temp_hyperparam": self.t, 
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "out_dim": self.out_dim,
            "weight_decay": self.weight_decay,
            "learning_rate": self.learning_rate
        })
        
        model = TaUSimCLR(device=self.device, 
            t=self.t, eps=self.eps,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size, 
            confidence_threshold=self.confidence_threshold, 
            out_dim=self.out_dim, 
            backbone_name=self.backbone_name,
            use_final_bn=self.use_final_bn, 
            use_dropout=self.use_dropout,
            learning_rate=self.learning_rate
        )

        #if torch.cuda.device_count() > 1:
        #    print("using", torch.cuda.device_count(), "GPUs...")
        #    model.backbone = nn.DataParallel(model.backbone)
        #    model.classifier = nn.DataParallel(model.classifier)

        wandb.watch(model, log="all")

        optimizer = optim.Adam(model.parameters(), 
                               weight_decay=self.weight_decay,
                               lr=self.learning_rate)
        finetuner = optim.Adam(model.classifier.parameters(), 
                              weight_decay=self.weight_decay,
                              lr=self.learning_rate)

        def train(model, device, train_loader, optimizer, epoch):
            model.train()
            losses = []
            for i, batch in enumerate(tqdm(train_loader, desc='Training')):
                X1, X2 = batch[0]
                X1 = X1.to(device)
                X2 = X2.to(device)
                z1, temp1 = model(X1)
                z2, temp2 = model(X2)
                optimizer.zero_grad()
                loss = model.tau_loss(z1, z2, temp1, temp2)
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
                z1, temp1 = model(X1)
                z2, temp2 = model(X2)
                loss = model.tau_loss(z1, z2, temp1, temp2)
                wandb.log({"val_loss": loss})
                losses.append(loss.detach())
            avg_loss = sum(losses) / len(losses)
            wandb.log({"per_epoch_val_loss": avg_loss, "train_epoch": epoch})

        def finetune(model, device, finetune_loader, optimizer, epoch):
            preds = []
            trues = []
            trues_val = []
            preds_val = []
            taus = []
            for i, batch in enumerate(tqdm(finetune_loader, desc='Finetuning')):
                feats = model.get_features(batch[0].to(device))
                feats = torch.squeeze(feats)
                H, tau = feats[:, :-1], feats[:, -1].unsqueeze(1)
                y = batch[1].to(device)
                z = model.classifier(H)
                finetuner.zero_grad()
                loss = model.clf_loss(z, y.long())
                loss.backward()
                finetuner.step()

                trues += y.detach().to('cpu').numpy().tolist()
                preds += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()
                taus += tau.detach().to('cpu').numpy().tolist()

            for k, batch in enumerate(tqdm(finetune_val_loader, desc='Finetuning')):
                feats = model.get_features(batch[0].to(device))
                feats = torch.squeeze(feats)
                H, tau = feats[:, :-1], feats[:, -1].unsqueeze(1)
                y = batch[1].to(device)
                z = model.classifier(H)
                val_loss = model.clf_loss(z, y.long())

                trues_val += y.detach().to('cpu').numpy().tolist()
                preds_val += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()
                                        
            ft_acc = accuracy_score(trues, preds)
            ft_val_acc = accuracy_score(trues_val, preds_val)
            self.tau_stats["mean"] = np.mean(taus)
            self.tau_stats["max"] = np.max(taus)
            self.tau_stats["stdev"] = np.std(taus) 
            self.tau_stats["min"] = np.min(taus)
            wandb.log({"ft_acc": ft_acc, "ft_epoch": epoch})
            wandb.log({"ft_val_acc": ft_val_acc, "ft_epoch": epoch})
            wandb.log({"mean_tau": self.tau_stats["mean"], "stdev_tau": self.tau_stats["stdev"],
                      "max_tau": self.tau_stats["max"], "min_tau": self.tau_stats["min"]})

        def test(model, device, test_loader, confidence_t, best_model=None, best_clf=None, log=True):
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
                    h, tau = feats[:, :-1], feats[:, -1].unsqueeze(1)
                    tau = 1 / tau
                    tau = (tau - tau.min()) / (tau.max() - tau.min())
                    includes = torch.Tensor((tau.cpu().numpy() <= confidence_t).astype(int)).squeeze()
                    indices = torch.nonzero(includes)
                    logits = model.classifier(h.squeeze())
                    print(torch.argmax(logits, dim=-1))
                    print(label)
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
            print(test_metrics)
            return test_metrics, len(filtered_cores)

        def plot_accuracy_vs_confidence_curve(model, device, test_loader, best_model=None, best_clf=None):
            df = pd.DataFrame()
            for thresh in range(0,101,10):
                t = thresh / 100
                metrics, remaining_cores = test(model, device, test_loader, t, best_model, best_clf, log=False)
                log_df = {"confidence_threshold": thresh, "remaining_cores": remaining_cores}
                for key in metrics:
                    new_key = key.replace("test/", "")
                    log_df[new_key] = metrics[key]

                print('cauroc', log_df['core_auroc'])
                print('pauroc', log_df['patch_auroc'])
    
                df = df.append(log_df, ignore_index=True)
            
            df.to_csv("../../../../tau_vs_perf_df.csv")

        for epoch in range(1+self.train_epochs):
            train(model, self.device, train_loader, optimizer, epoch)
            validate(model, self.device, val_loader, optimizer, epoch)
            torch.save(model.state_dict(), CHECKPOINTS_DIR_MODEL.format(epoch))
        
        for epoch in range(1+self.finetune_epochs):
            finetune(model, self.device, finetune_loader, optimizer, epoch)
            torch.save(model.classifier.state_dict(), CHECKPOINTS_DIR_CLF.format(epoch))
        
        test(model, self.device, test_loader, self.confidence_threshold, best_model='_tau_256_best_model', best_clf='_tau_256_clf_epoch49')

        plot_accuracy_vs_confidence_curve(model, self.device, test_loader, best_model='_tau_256_best_model', best_clf='_tau_256_clf_epoch49')
        
        wandb.alert(
            title='Run complete',
            text=f'Your TaU+SimCLR model is done training! Check out https://wandb.ai/home for details.',
            level=wandb.AlertLevel.INFO,
        )

    