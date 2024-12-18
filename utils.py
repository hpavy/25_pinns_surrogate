# Les fonctions utiles ici
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from model import PINNs
import numpy as np
import torch


def write_csv(data, path, file_name):
    dossier = Path(path)
    df = pd.DataFrame(data)
    # Créer le dossier si il n'existe pas
    dossier.mkdir(parents=True, exist_ok=True)
    df.to_csv(path + file_name)


def read_csv(path):
    return pd.read_csv(path)


def charge_data(hyper_param, param_adim):
    """
    Charge the data of X_full, U_full with every points
    And X_train, U_train with less points
    """
    # La data
    # On adimensionne la data
    nb_simu = len(hyper_param["file"])
    df_tot = pd.DataFrame()
    for k in range(nb_simu):
        df = pd.read_csv("25_pinns_surrogate/" + hyper_param["file"][k])
        # df = pd.read_csv(hyper_param["file"][k])
        df_modified = df.loc[
            (df["Points:0"] >= hyper_param["x_min"])
            & (df["Points:0"] <= hyper_param["x_max"])
            & (df["Points:1"] >= hyper_param["y_min"])
            & (df["Points:1"] <= hyper_param["y_max"])
            & (df["Time"] > hyper_param["t_min"])
            & (df["Time"] < hyper_param["t_max"])
            & (df["Points:2"] == 0.0)
            # pour ne pas avoir dans le cylindre
            & (df["Points:0"] ** 2 + df["Points:1"] ** 2 > (0.025 / 2) ** 2),
            :,
        ].copy()
        df_modified.loc[:, "ya0"] = hyper_param["ya0"][k]
        df_tot = pd.concat([df_tot, df_modified])
        print(f"fichier n°{k} chargé")

    # Adimensionnement
    x_full, y_full, t_full, ya0_full = (
        torch.tensor(df_tot["Points:0"], dtype=torch.float32) / param_adim["L"],
        torch.tensor(df_tot["Points:1"], dtype=torch.float32) / param_adim["L"],
        torch.tensor(df_tot["Time"], dtype=torch.float32)
        / (param_adim["L"] / param_adim["V"]),
        torch.tensor(df_tot["ya0"], dtype=torch.float32) / param_adim["L"],
    )
    u_full, v_full, p_full = (
        torch.tensor(df_tot["Velocity:0"], dtype=torch.float32) / param_adim["V"],
        torch.tensor(df_tot["Velocity:1"], dtype=torch.float32) / param_adim["V"],
        torch.tensor(df_tot["Pressure"], dtype=torch.float32)
        / ((param_adim["V"] ** 2) * param_adim["rho"]),
    )

    # Normalisation Z
    x_norm_full = (x_full - x_full.mean()) / x_full.std()
    y_norm_full = (y_full - y_full.mean()) / y_full.std()
    t_norm_full = (t_full - t_full.mean()) / t_full.std()
    ya0_norm_full = (ya0_full - ya0_full.mean()) / ya0_full.std()
    p_norm_full = (p_full - p_full.mean()) / p_full.std()
    u_norm_full = (u_full - u_full.mean()) / u_full.std()
    v_norm_full = (v_full - v_full.mean()) / v_full.std()

    X_full = torch.stack((x_norm_full, y_norm_full, t_norm_full, ya0_norm_full), dim=1)
    U_full = torch.stack((u_norm_full, v_norm_full, p_norm_full), dim=1)

    x_int = (x_norm_full.max() - x_norm_full.min()) / hyper_param["nb_points_axes"]
    y_int = (y_norm_full.max() - y_norm_full.min()) / hyper_param["nb_points_axes"]
    X_train = torch.zeros((0, 4))
    U_train = torch.zeros((0, 3))
    print("Starting X_train")
    for time in torch.unique(t_norm_full):
        # les points autour du cylindre dans un rayon de 0.025
        masque = ((x_full**2 + y_full**2) < ((0.025 / param_adim["L"]) ** 2)) & (
            t_norm_full == time
        )
        indices = torch.randperm(len(x_norm_full[masque]))[
            :, hyper_param["nb_points_close_cylinder"]
        ]

        new_x = torch.stack(
            (
                x_norm_full[masque][indices],
                y_norm_full[masque][indices],
                t_norm_full[masque][indices],
                ya0_norm_full[masque][indices],
            ),
            dim=1,
        )
        new_y = torch.stack(
            (
                u_norm_full[masque][indices],
                v_norm_full[masque][indices],
                p_norm_full[masque][indices],
            ),
            dim=1,
        )
        X_train = torch.cat((X_train, new_x))
        U_train = torch.cat((U_train, new_y))

        # Les points avec 'latin hypercube sampling'
        for x_num in range(hyper_param["nb_points_axes"]):
            for y_num in range(hyper_param["nb_points_axes"]):
                masque = (
                    (x_norm_full > x_norm_full.min() + x_int * x_num)
                    & (x_norm_full < x_norm_full.min() + (x_num + 1) * x_int)
                    & (y_norm_full < y_norm_full.min() + (y_num + 1) * y_int)
                    & (y_norm_full > y_norm_full.min() + (y_num) * y_int)
                    & (t_norm_full == time)
                )
                if len(x_norm_full[masque]) > 0:
                    indice = torch.randint(x_norm_full[masque].size[0], (1,)).item()
                    new_x = torch.tensor(
                        [
                            x_norm_full[masque][indice],
                            y_norm_full[masque][indice],
                            t_norm_full[masque][indice],
                            ya0_norm_full[masque][indice],
                        ]
                    ).reshape(-1, 4)
                    new_y = torch.tensor(
                        [
                            u_norm_full[masque][indice],
                            v_norm_full[masque][indice],
                            p_norm_full[masque][indice],
                        ]
                    ).reshape(-1, 3)
                    X_train = torch.cat((X_train, new_x))
                    U_train = torch.cat((U_train, new_y))
    indices = torch.randperm(X_train.shape(0))
    X_train = X_train[indices]
    U_train = U_train[indices]
    print("X_train OK")

    # les points du bord
    nb_border = hyper_param["nb_points_border"]
    teta_int = np.linspace(0, 2 * np.pi, nb_border)
    X_border = np.zeros((0, 4))
    for time in np.unique(t_norm_full):
        for ya0_ in hyper_param["ya0"][k]:
            for teta in teta_int:
                x_ = (
                    (((0.025 / 2) * np.cos(teta)) / param_adim["L"]) - x_full.mean()
                ) / x_full.std()
                y_ = (
                    (((0.025 / 2) * np.sin(teta)) / param_adim["L"]) - y_full.mean()
                ) / y_full.std()
                new_x = np.array([x_, y_, time, ya0_]).reshape(-1, 4)
                X_border = torch.cat((X_border, new_x))
    indices = np.random.permutation(len(X_border))
    X_border = X_border[indices]
    print("X_border OK")

    teta_int_test = np.linspace(0, 2 * np.pi, 15)
    X_border_test = np.zeros((0, 4))
    for time in np.unique(t_norm_full):
        for ya0_ in hyper_param["ya0"][k]:
            for teta in teta_int_test:
                x_ = (
                    (((0.025 / 2) * np.cos(teta)) / param_adim["L"]) - x_full.mean()
                ) / x_full.std()
                y_ = (
                    (((0.025 / 2) * np.sin(teta)) / param_adim["L"]) - y_full.mean()
                ) / y_full.std()
                new_x = np.array([x_, y_, time, ya0_]).reshape(-1, 4)
                X_border_test = np.concatenate((X_border_test, new_x))

    # les valeurs pour renormaliser ou dénormaliser
    mean_std = {
        "u_mean": u_full.mean(),
        "v_mean": v_full.mean(),
        "p_mean": p_full.mean(),
        "x_mean": x_full.mean(),
        "y_mean": y_full.mean(),
        "t_mean": t_full.mean(),
        "x_std": x_full.std(),
        "y_std": y_full.std(),
        "t_std": t_full.std(),
        "u_std": u_full.std(),
        "v_std": v_full.std(),
        "p_std": p_full.std(),
        "ya0_mean": ya0_full.mean(),
        "ya0_std": ya0_full.std(),
    }

    return X_train, U_train, X_full, U_full, X_border, X_border_test, mean_std


def init_model(f, hyper_param, device, folder_result):
    model = PINNs(hyper_param).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyper_param["lr_init"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hyper_param["gamma_scheduler"]
    )
    loss = nn.MSELoss()
    # On regarde si notre modèle n'existe pas déjà
    if Path(folder_result + "/model_weights.pth").exists():
        # Charger l'état du modèle et de l'optimiseur
        checkpoint = torch.load(folder_result + "/model_weights.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        weights = checkpoint["weights"]
        print("\nModèle chargé\n", file=f)
        print("\nModèle chargé\n")
        csv_train = read_csv(folder_result + "/train_loss.csv")
        csv_test = read_csv(folder_result + "/test_loss.csv")
        train_loss = {
            "total": list(csv_train["total"]),
            "data": list(csv_train["data"]),
            "pde": list(csv_train["pde"]),
            "border": list(csv_train["border"]),
        }
        test_loss = {
            "total": list(csv_test["total"]),
            "data": list(csv_test["data"]),
            "pde": list(csv_test["pde"]),
            "border": list(csv_test["border"]),
        }
        print("\nLoss chargée\n", file=f)
        print("\nLoss chargée\n")
    else:
        print("Nouveau modèle\n", file=f)
        print("Nouveau modèle\n")
        train_loss = {"total": [], "data": [], "pde": [], "border": []}
        test_loss = {"total": [], "data": [], "pde": [], "border": []}
        weights = {
            "weight_data": hyper_param["weight_data"],
            "weight_pde": hyper_param["weight_pde"],
            "weight_border": hyper_param["weight_border"],
        }
    return model, optimizer, scheduler, loss, train_loss, test_loss, weights


if __name__ == "__main__":
    write_csv([[1, 2, 3], [4, 5, 6]], "ready_cluster/piche/test.csv")
