import torch
from pyDOE import lhs


class Rectangle:
    def __init__(self, x_max, y_max, t_min, t_max, x_min=0, y_min=0):
        """on crée ici un rectangle

        Args:
            x_max (_type_): la taille en x maximale
            y_max (_type_): la taille en y maximale
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max

    def generate_border(self, n):
        """génère n valeurs randoms sur les bords du rectangle"""
        if n % 4 != 0:
            raise ValueError("mettre n divisible par 4")  # A changer plus tard
        points = torch.zeros(n, 2)
        points[: n // 4] = torch.stack(
            (
                self.x_min * torch.ones(n // 4),
                (self.y_max - self.y_min) * torch.rand(n // 4),
            ),
            dim=1,
        )
        points[n // 4: n // 2] = torch.stack(
            (
                self.x_max * torch.ones(n // 4),
                (self.y_max - self.y_min) * torch.rand(n // 4),
            ),
            dim=1,
        )
        points[n // 2: 3 * n // 4] = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n // 4),
                self.y_max * torch.ones(n // 4),
            ),
            dim=1,
        )
        points[3 * n // 4:] = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n // 4),
                self.y_min * torch.ones(n // 4),
            ),
            dim=1,
        )
        return torch.cat(
            (points, torch.rand(n, 1) * (self.t_max - self.t_min)), dim=1
        ).requires_grad_()

    def generate_random(self, n, init=False):
        """génère n valeurs randoms dans le rectangle avec un temps aléatoire,
        si init est True, alors le temps est initialisé à 0
        """
        points = torch.stack(
            (
                (self.x_max - self.x_min) * torch.rand(n) + self.x_min,
                (self.y_max - self.y_min) * torch.rand(n) + self.y_min,
            ),
            dim=1,
        )
        if not init:
            return torch.cat(
                (points, torch.rand(n, 1) * (self.t_max - self.t_min) + self.t_min),
                dim=1,
            ).requires_grad_()
        else:
            return torch.cat((points, torch.zeros(n, 1)), dim=1).requires_grad_()

    def generate_lhs(self, n):
        """Donne répartition latin hypercube"""
        max_min = torch.tensor(
            [self.x_max - self.x_min, self.y_max -
                self.y_min, self.t_max - self.t_min]
        )
        minn = torch.tensor([self.x_min, self.y_min, self.t_min])
        tensor_final = minn + max_min * torch.from_numpy(lhs(3, n))
        return tensor_final.to(dtype=torch.float32).requires_grad_()


class RectangleWithoutCylinder:
    def __init__(
        self,
        x_max,
        y_max,
        t_min,
        t_max,
        x_cyl,
        y_cyl,
        r_cyl,
        mean_std,
        param_adim,
        x_min=0,
        y_min=0,
    ):
        """on crée ici un rectangle

        Args:
            x_max (_type_): la taille en x maximale
            y_max (_type_): la taille en y maximale
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.t_min = t_min
        self.t_max = t_max
        self.x_cyl = x_cyl
        self.y_cyl = y_cyl
        self.r_cyl = r_cyl
        self.mean_std = mean_std
        self.param_adim = param_adim

    def generate_lhs(self, n):
        """Donne répartition latin hypercube"""
        max_min = torch.tensor(
            [self.x_max - self.x_min, self.y_max -
                self.y_min, self.t_max - self.t_min]
        )
        minn = torch.tensor([self.x_min, self.y_min, self.t_min])
        tensor_final = torch.zeros((0, 3))
        while tensor_final.shape[0] < n:
            n_left = n - tensor_final.shape[0]
            test = minn + max_min * torch.from_numpy(lhs(3, n_left))
            test_x_dim = (
                test[:, 0] * self.mean_std["x_std"] + self.mean_std["x_mean"]
            ) * self.param_adim["L"]
            test_y_dim = (
                test[:, 1] * self.mean_std["y_std"] + self.mean_std["y_mean"]
            ) * self.param_adim["L"]
            test_good = test[
                ((test_x_dim - self.x_cyl) ** 2 + (test_y_dim - self.y_cyl) ** 2)
                > self.r_cyl**2
            ]
            tensor_final = torch.concatenate((tensor_final, test_good))
        return tensor_final.to(dtype=torch.float32).requires_grad_()


if __name__ == "__main__":
    test = Rectangle(x_max=5, y_max=2, t_min=0, t_max=15)
    print(test.generate_lhs(96).dtype)
