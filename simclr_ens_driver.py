import torch

from src.modeling.registry import create_model
from src.typing import FeatureExtractionProtocol
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

CHECKPOINTS_DIR_MODEL = '../../../../simclr/{0}.pt'
CHECKPOINTS_DIR_ENSEMBLE = '../../../../simclr_ensemble/{0}_{1}.pt'
CHECKPOINTS_DIR_CLF = '../../../../simclr_ensemble_clf/{0}.pt'
CHECKPOINTS_DIR_ENSEMBLE_CLF = '../../../../simclr_ensemble_clf/{0}_{1}.pt'

class SimCLREnsDriver:
    def __init__(self, device, batch_size, 
                 train_epochs, finetune_epochs,
                 t, eps, out_dim, hidden_dim,
                 use_final_bn, use_dropout, 
                 learning_rate, use_ensembles,
                 ensemble_models,
                 backbone_name, datamodule,
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

        self.use_ensembles = use_ensembles
        self.ensemble_models = ensemble_models

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
            def __init__(self, device, batch_size, use_ensembles,
                 t, eps, out_dim, hidden_dim, learning_rate, use_final_bn,
                 ensemble_dim, use_dropout, backbone_name):
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
            use_ensembles=self.use_ensembles,
            ensemble_dim=self.ensemble_models,
            use_dropout=self.use_dropout,
            learning_rate=self.learning_rate
        )

        if torch.cuda.device_count() > 1:
            print("using", torch.cuda.device_count(), "GPUs...")
            self.features_dim = model.backbone.fc.in_features
            model.backbone = nn.DataParallel(model.backbone)
            model.classifier = nn.DataParallel(model.classifier)

        optimizers = []
        finetuners = []
        ensemble_model = []
        if self.use_ensembles:
            for _ in range(self.ensemble_models):
                finetuners.append(optim.Adam(model.parameters(), weight_decay=1e-6,
                                    lr=self.learning_rate))
        else:
            torch.manual_seed(88)
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

        def train_ensemble(ensemble, device, val_loader, optimizer, epoch):
            for i in range(len(ensemble)):
                model = ensemble[i]
                model.train()
                losses = []
                for j, batch in enumerate(tqdm(train_loader, desc='Training')):
                    X1, X2 = batch[0]
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    z1 = model(X1)
                    z2 = model(X2)
                    optimizers[i].zero_grad()
                    loss = model.NT_Xent(z1, z2)
                    loss.backward()
                    optimizers[i].step()
                    wandb.log({"train_loss": loss})
                    losses.append(loss.detach())
                avg_loss = sum(losses) / len(losses)
                wandb.log({"model{0}_avg_train_loss".format(i): avg_loss, 
                           "train_epoch": epoch})

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

        def validate_ensemble(ensemble, device, val_loader, optimizer, epoch):
            for i in range(len(ensemble)):
                model = ensemble[i]
                model.eval()
                losses = []
                for j, batch in enumerate(tqdm(val_loader, desc='Validation')):
                    X1, X2 = batch[0]
                    X1 = X1.to(device)
                    X2 = X2.to(device)
                    z1 = model(X1)
                    z2 = model(X2)
                    loss = model.NT_Xent(z1, z2)
                    wandb.log({"val_loss": loss})
                    losses.append(loss.detach())
                avg_loss = sum(losses) / len(losses)
                wandb.log({"model{0}_avg_val_loss".format(i): avg_loss, 
                           "train_epoch": epoch})

        def finetune(model, device, finetune_loader, optimizer, epoch):
            model.eval()
            preds = []
            trues = []
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
            
            ft_acc = accuracy_score(trues, preds)
            print("[Epoch {0}] Finetune Accuracy Score: {1}".format(epoch, ft_acc))
            wandb.log({"ft_acc": ft_acc, "ft_epoch": epoch})

        def finetune_ensemble(model, clf, device, finetune_loader, optimizer, criterion, epoch):
            model.eval()
            clf.train()
            preds = []
            trues = []
            preds_val = []
            trues_val = []
            for j, batch in enumerate(tqdm(finetune_loader, desc='Finetuning (Training)')):
                H = model.get_features(batch[0].to(device))
                y = batch[1].to(device)
                z = clf(H.squeeze())
                optimizer.zero_grad()
                loss = criterion(z, y.long())
                loss.backward()
                optimizer.step()
                wandb.log({"ft_loss_clf_{0}".format(i): loss})

                trues += y.detach().to('cpu').numpy().tolist()
                preds += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()

            for k, batch in enumerate(tqdm(finetune_val_loader, desc='Finetuning (Validation)')):
                H = model.get_features(batch[0].to(device))
                y = batch[1].to(device)
                z = clf(H.squeeze())
                val_loss = criterion(z, y.long())
                wandb.log({"ft_val_loss_clf_{0}".format(i): val_loss})

                trues_val += y.detach().to('cpu').numpy().tolist()
                preds_val += np.argmax(z.detach().to('cpu').numpy(), axis=1).tolist()
            
            ft_acc = accuracy_score(trues, preds)
            ft_val_acc = accuracy_score(trues_val, preds_val)
            wandb.log({"ft_acc_clf_{0}".format(i): ft_acc, "ft_epoch": epoch})
            wandb.log({"ft_val_acc_clf_{0}".format(i): ft_val_acc, "ft_epoch": epoch})

        def test(model, device, test_loader, best_model=None, best_clf=None, log=True):
            model.eval()
            
            if best_model:
                model.load_state_dict(torch.load(CHECKPOINTS_DIR_MODEL.format(best_model)))
            if best_clf:
                #model.classifier.load_state_dict(torch.load(CHECKPOINTS_DIR_CLF.format(best_clf)))
                features_dim = model.features_dim
                model.classifier = nn.Sequential(
                    nn.Linear(features_dim, features_dim // 2),
                    nn.ReLU(),
                    nn.Linear(features_dim // 2, 2),
                    nn.ReLU()
                ).to(self.device)
                model.classifier.load_state_dict(torch.load(CHECKPOINTS_DIR_CLF.format(best_clf)))

            from src.utils.metrics import PatchwiseAndCorewiseMetrics
            metrics_fn = PatchwiseAndCorewiseMetrics(
                compute_corewise_and_patchwise=True, 
                breakdown_by_center=False, 
                metrics=['auroc', 'acc_macro', 'avg_prec']
            )

            from src.utils.metrics import OutputCollector
            output_collector = OutputCollector()

            for patch, label, metadata in tqdm(test_loader):
                with torch.inference_mode():
                    feats = model.get_features(patch.to(device))
                    feats = torch.squeeze(feats)
                    pos = metadata["position"]
                    logits = model.classifier(feats)
                output_collector.collect_batch({
                    'logits':logits,
                    'labels': label,
                    'position': pos, **metadata
                })

            out = output_collector.compute()
            output_collector.reset()

            test_metrics = {}
            metric_fn_out = metrics_fn(out)
            for metric_name in metric_fn_out:
                test_metrics['test/' + metric_name] = metric_fn_out[metric_name].item()
            
            if log:
                wandb.log(test_metrics)

            return test_metrics

        def test_ensemble(model, ensemble, device, test_loader, confidence_t=1.0, log=True, best_model=None, best_clfs=None):
            model.eval()

            from src.utils.metrics import PatchwiseAndCorewiseMetrics, OutputCollector
            metrics_fn = PatchwiseAndCorewiseMetrics(
                compute_corewise_and_patchwise=True, 
                breakdown_by_center=False, 
                metrics=['auroc', 'acc_macro', 'avg_prec', 'brier_score', 'expected_calibration_error']
            )
            
            for i in range(len(ensemble)):            
                if best_model:
                    model.load_state_dict(torch.load(CHECKPOINTS_DIR_MODEL.format(best_model)))
                if best_clfs:
                    ensemble[i].load_state_dict(torch.load(CHECKPOINTS_DIR_ENSEMBLE_CLF.format(best_clfs[i], i)))

            output_collector = OutputCollector()

            for patch, label, metadata in tqdm(test_loader):
                with torch.inference_mode():
                    feats = model.get_features(patch.to(device))
                    feats = torch.squeeze(feats)
                    ens_logits = torch.concat([ensemble[i](feats).unsqueeze(-1) for i in range(self.ensemble_models)], dim=-1)
                    pos = metadata["position"]

                    avg_logits = torch.mean(ens_logits, dim=-1)
                    ens_preds = torch.argmax(ens_logits, dim=-1).float()
                    avg_preds = ens_preds.mean(-1).int().float()
                    u = ens_preds.var(-1)
                    uncertainty_scores = (u - u.min()) / (u.max() - u.min())

                    print(uncertainty_scores)
                    
                    all_cores = metadata["core_specifier"]
                    includes = torch.Tensor((uncertainty_scores.cpu().numpy() <= confidence_t).astype(int)).squeeze()
                    indices = torch.nonzero(includes)
                    filtered_cores = [all_cores[i] for i in range(len(includes)) if includes[i] != 0]
                
                output_collector.collect_batch({
                    'logits': avg_logits[indices].squeeze(1), 
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
        
        def plot_acc_vs_confidence(model, ensemble, test_loader, best_model, best_clfs):
            import pandas as pd
            df = pd.DataFrame()
            for kappa in range(0, 11):
                kappa = kappa / 10
                metrics, remaining_cores = test_ensemble(model, ensemble, self.device, test_loader, confidence_t=kappa, best_model=best_model, 
                                                         best_clfs=best_clfs, log=False)
                log_df = {"confidence_threshold": kappa, "num_ens": len(best_clfs), "remaining_cores": remaining_cores}
                for key in metrics:
                    new_key = key.replace("test/", "")
                    log_df[new_key] = metrics[key]
                df = df.append(log_df, ignore_index=True)
                
            df.to_csv("../../../../ens_vs_perf_df.csv")

        #for epoch in range(self.train_epochs):
        #    if self.use_ensembles:
        #        train_ensemble(ensemble_model, self.device, train_loader, optimizer, epoch)
        #        validate_ensemble(ensemble_model, self.device, val_loader, optimizer, epoch)
        #        for i in range(len(ensemble_model)):
        #            model = ensemble_model[i]
        #            torch.save(model.state_dict(), CHECKPOINTS_DIR_ENSEMBLE.format(epoch, i))
        #    
        #    else:
        #       train(model, self.device, train_loader, optimizer, epoch)
        #       validate(model, self.device, val_loader, optimizer, epoch)
        #       torch.save(model.state_dict(), CHECKPOINTS_DIR_MODEL.format(epoch))

        #train(model, self.device, train_loader, optimizer, epoch)
        #validate(model, self.device, val_loader, optimizer, epoch)
        #torch.save(model.state_dict(), CHECKPOINTS_DIR_ENSEMBLE.format(epoch, 3))
        # 

        if self.use_ensembles:
            ensemble = []
            for idx in range(self.ensemble_models):
                features_dim = self.features_dim
                seed = torch.randint(0,999,(1,)).item()
                torch.manual_seed(seed)
                wandb.log({"model": idx, "seed": seed})
                mlp = nn.Sequential(
                    nn.Linear(features_dim, features_dim // 2),
                    nn.LeakyReLU(),
                    nn.Linear(features_dim // 2, 2),
                    nn.LeakyReLU()
                ).to(self.device)
                ensemble.append(mlp)

            i = 0
            for mlp in ensemble:
                clfcriterion = nn.CrossEntropyLoss()
                optimiz = optim.Adam(mlp.parameters(), lr=1e-3)
                for epoch in range(self.finetune_epochs):
                    finetune_ensemble(model, mlp, self.device, finetune_loader, optimiz, clfcriterion, epoch)
                    torch.save(mlp.state_dict(), CHECKPOINTS_DIR_ENSEMBLE_CLF.format(epoch, i))
                i += 1

            torch.manual_seed(88)
            test_ensemble(model, ensemble, self.device, test_loader, best_model=199, 
                          best_clfs=['__14', '__17', '__11'])

            plot_acc_vs_confidence(model, ensemble, test_loader, best_model=199, best_clfs=['__14', '__17', '__11'])
        
        else:
            test(model, self.device, test_loader, best_model=199, best_clf='14_0')
        
        wandb.alert(
            title='Run complete',
            text=f'Your SimCLR model is done training! Check out https://wandb.ai/home for details.',
            level=wandb.AlertLevel.INFO,
        )

    