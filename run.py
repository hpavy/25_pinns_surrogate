from geometry import RectangleWithoutCylinder
import torch
from utils import charge_data, init_model
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json


class RunSimulation:
    def __init__(self, hyper_param, folder_result_name, param_adim):
        self.hyper_param = hyper_param
        self.time_start = time.time()
        self.folder_result_name = folder_result_name
        self.folder_result = "results/" + folder_result_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_adim = param_adim
        self.nb_simu = len(self.hyper_param["file"])

    def run(self):
        # Charging the model
        # Creation du dossier de result
        Path(self.folder_result).mkdir(parents=True, exist_ok=True)
        if not Path(self.folder_result + "/hyper_param.json").exists():
            with open(self.folder_result + "/hyper_param.json", "w") as file:
                json.dump(self.hyper_param, file, indent=4)
            self.hyper_param = self.hyper_param

        else:
            with open(self.folder_result + "/hyper_param.json", "r") as file:
                self.hyper_param = json.load(file)

        # Data loading
        (
            X_train_np,
            U_train_np,
            X_full,
            U_full,
            X_border_np,
            X_border_test_np,
            mean_std,
        ) = charge_data(self.hyper_param, self.param_adim)
        X_train = torch.from_numpy(X_train_np).requires_grad_().to(torch.float32)
        U_train = torch.from_numpy(U_train_np).requires_grad_().to(torch.float32)
        X_border = torch.from_numpy(X_border_np).requires_grad_().to(torch.float32)
        X_border_test = (
            torch.from_numpy(X_border_test_np).requires_grad_().to(torch.float32)
        )

        # le domaine de résolution
        rectangle = RectangleWithoutCylinder(
            x_max=X_full[:, 0].max(),
            y_max=X_full[:, 1].max(),
            t_min=X_full[:, 2].min(),
            t_max=X_full[:, 2].max(),
            x_min=X_full[:, 0].min(),
            y_min=X_full[:, 1].min(),
            x_cyl=0.0,
            y_cyl=0.0,
            r_cyl=0.025 / 2,
            mean_std=mean_std,
            param_adim=self.param_adim,
        )

        X_pde = torch.empty(
            self.hyper_param["nb_points_pde"] * self.nb_simu, requires_grad=True
        )
        for k in range(self.nb_simu):
            X_pde_without_param = torch.concat(
                rectangle.generate_lhs(self.hyper_param["nb_points_pde"]),
                self.ya0[k] * torch.ones(self.hyper_param["nb_points_pde"]),
            )
            X_pde[
                k
                * self.hyper_param["nb_points_pde"] : (k + 1)
                * self.hyper_param["nb_points_pde"]
            ] = X_pde_without_param
        indices = torch.randperm(X_pde.size(0))
        X_pde = X_pde[indices, :]

        # Data test loading
        X_test_pde = torch.empty(
            self.hyper_param["n_pde_test"] * self.nb_simu, requires_grad=True
        )
        for k in range(self.nb_simu):
            X_test_pde_without_param = rectangle.generate_lhs(
                self.hyper_param["n_pde_test"]
            )
            X_test_pde[
                k
                * self.hyper_param["n_pde_test"] : (k + 1)
                * self.hyper_param["n_pde_test"]
            ] = X_test_pde_without_param
        indices = torch.randperm(X_test_pde.size(0))
        X_test_pde = X_test_pde[indices, :]

        points_coloc_test = np.random.choice(
            len(X_full), self.hyper_param["n_data_test"], replace=False
        )
        X_test_data = torch.from_numpy(X_full[points_coloc_test])
        U_test_data = torch.from_numpy(U_full[points_coloc_test])

        # Initialiser le modèle

        # On plot les print dans un fichier texte
        with open(self.folder_result + "/print.txt", "a") as f:
            model, optimizer, scheduler, loss, train_loss, test_loss, weights = (
                init_model(f, self.hyper_param, self.device, self.folder_result)
            )
            # On entraine le modèle
            if self.hyper_param["dynamic_weights"]:
                weight_data = weights["weight_data"]
                weight_pde = weights["weight_pde"]
                weight_border = weights["weight_border"]
            else:
                weight_data = self.hyper_param["weight_data"]
                weight_pde = self.hyper_param["weight_pde"]
                weight_border = self.hyper_param["weight_border"]
            train(
                nb_epoch=self.hyper_param["nb_epoch"],
                train_loss=train_loss,
                test_loss=test_loss,
                weight_data_init=weight_data,
                weight_pde_init=weight_pde,
                weight_border_init=weight_border,
                dynamic_weights=self.hyper_param["dynamic_weights"],
                lr_weights=self.hyper_param["lr_weights"],
                model=model,
                loss=loss,
                optimizer=optimizer,
                X_train=X_train,
                U_train=U_train,
                X_pde=X_pde,
                X_test_pde=X_test_pde,
                X_test_data=X_test_data,
                U_test_data=U_test_data,
                Re=self.hyper_param["Re"],
                time_start=self.time_start,
                f=f,
                folder_result=self.folder_result,
                save_rate=self.hyper_param["save_rate"],
                batch_size=self.hyper_param["batch_size"],
                scheduler=scheduler,
                X_border=X_border,
                X_border_test=X_border_test,
                mean_std=mean_std,
                w_0=(self.hyper_param["H"] / self.hyper_param["m"]) ** 0.5,
                param_adim=self.param_adim,
            )
