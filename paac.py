import torch.nn as nn
from torch.autograd import Variable
from PIL import Image
from PIL.Image import BILINEAR
from torchvision.transforms import ToTensor, ToPILImage

INPUT_CHANNELS = 4
INPUT_IMAGE_SIZE = (84, 84)


class PAACNet(nn.Module):
    to_tensor = ToTensor()
    to_pil_image = ToPILImage()

    def __init__(self, num_actions):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU()
        )

        self.fc = nn.Linear(3136, 512)

        self.policy_output = nn.Sequential(
            nn.Linear(512, num_actions),
            nn.Softmax(1)
        )

        self.value_output = nn.Linear(512, 1)

        # init weights and biases
        import torch.nn.init as init

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    @classmethod
    def preprocess(cls, x):
        r"""preprocesses & converts the output of gym environment

        :param x: grayscale array with shape (210, 160, 1)
        :return: preprocessed & converted tensor
        """

        # TODO : support flickering games by picking max pixels
        x = Image.fromarray(x.squeeze(), 'L')
        x = x.resize(INPUT_IMAGE_SIZE, resample=BILINEAR)
        return cls.to_tensor(x)

    def forward(self, x):
        r"""calculates PAAC outputs

        :param x: preprocessed states with shape (N, H, W, C)
        :return: tuple (policy_output, value_output)
        """
        x = self.conv_layers(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.policy_output(x), self.value_output(x)

    def policy(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.policy_output(x)

    def value(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.value_output(x)

    @staticmethod
    def entropy(x, epsilon=0):
        r"""calculates entropy

        :param x: policy_output with shape (N, L) where L is NUM_ACTIONS
        :param epsilon: epsilon for numerical stability
        :return: entropy
        """
        return -(x * (x + epsilon).log()).sum(0).sum(0)

    @staticmethod
    def log_and_negated_entropy(x, epsilon):
        log_x = (x + epsilon).log()
        return log_x, (x * log_x).sum(0).sum(0)

    @staticmethod
    def get_loss(q_values, values, log_a):
        r"""calculates policy loss and value loss

        :param q_values: Tensor with shape (T, N)
        :param values: Variable with shape (T, N)
        :param log_a: Variable with shape (T, N)
        :return: tuple (policy_loss, value_loss)
        """
        diff = Variable(q_values) - values

        # policy loss
        loss_p = -(Variable(diff.data) * log_a).mean(0)
        # value loss
        # 2 * nn.MSELoss
        double_loss_v = diff.pow(2).mean(0)
        loss = loss_p + 0.25 * double_loss_v
        return loss_p, double_loss_v, loss
