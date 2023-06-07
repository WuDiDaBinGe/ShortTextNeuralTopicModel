import torch
from torch import nn
from modeling_utils import BaseTopicNeuralModel

class WLDA(BaseTopicNeuralModel):
    def __init__(self, config):
        super(BaseTopicNeuralModel, self).__init__(config)
        

    def forward(self, bows):
        batch_size = bows.shape[0]
        normal_src_bow = bows / (torch.sum(bows, dim=1, keepdim=True) + 1e-8)
        latent_topics_logits = self._encode(normal_src_bow)
        latent_topics = F.softmax(latent_topics_logits, dim=1)
        bows_recon = self.decoder(latent_topic)
        # get _assist_loss
        theta_prior = self._sample(dist=self.dist, batch_size=batch_size, ori_data=bows)
        mmd = self._assist_loss(latent_topic, theta_prior, t=0.1)

        # get recon loss
        log_softmax_rec = torch.log_softmax(bows_recon, dim=1)

        rec_loss = -1.0 * torch.sum(bows * log_softmax_rec)

        s = torch.sum(bows) / len(bows)
        lamb = (5.0 * s * torch.log(torch.tensor(1.0 * bows.shape[-1])) / torch.log(torch.tensor(2.0)))
        mmd = mmd * lamb
        return bows_recon, latent_topics, rec_loss, mmd


    def _assist_loss(self, x, y, t=0.1, kernel='diffusion'):
        """
        computes the mmd loss with information diffusion kernel
        :param x: batch_size * latent dimension
        :param y: batch_size * latent dimension 先验分布
        :param t:
        :return:
        """
        eps = 1e-6
        n, d = x.shape
        if kernel == 'tv':
            sum_xx = torch.zeros(1).to(self.device)
            for i in range(n):
                for j in range(i + 1, n):
                    sum_xx = sum_xx + torch.norm(x[i] - x[j], p=1).to(self.device)
            sum_xx = sum_xx / (n * (n - 1))

            sum_yy = torch.zeros(1).to(self.device)
            for i in range(y.shape[0]):
                for j in range(i + 1, y.shape[0]):
                    sum_yy = sum_yy + torch.norm(y[i] - y[j], p=1).to(self.device)
            sum_yy = sum_yy / (y.shape[0] * (y.shape[0] - 1))

            sum_xy = torch.zeros(1).to(self.device)
            for i in range(n):
                for j in range(y.shape[0]):
                    sum_xy = sum_xy + torch.norm(x[i] - y[j], p=1).to(self.device)
            sum_yy = sum_yy / (n * y.shape[0])
        else:
            qx = torch.sqrt(torch.clamp(x, eps, 1))
            qy = torch.sqrt(torch.clamp(y, eps, 1))
            xx = torch.matmul(qx, qx.t())
            yy = torch.matmul(qy, qy.t())
            xy = torch.matmul(qx, qy.t())

            def diffusion_kernel(a, tmpt, dim):
                # return (4 * np.pi * tmpt)**(-dim / 2) * nd.exp(- nd.square(nd.arccos(a)) / tmpt)
                return torch.exp(-torch.acos(a).pow(2)) / tmpt

            off_diag = 1 - torch.eye(n).to(self.device)
            k_xx = diffusion_kernel(torch.clamp(xx, 0, 1 - eps), t, d - 1)
            k_yy = diffusion_kernel(torch.clamp(yy, 0, 1 - eps), t, d - 1)
            k_xy = diffusion_kernel(torch.clamp(xy, 0, 1 - eps), t, d - 1)
            sum_xx = (k_xx * off_diag).sum() / (n * (n - 1))
            sum_yy = (k_yy * off_diag).sum() / (n * (n - 1))
            sum_xy = 2 * k_xy.sum() / (n * n)
        return sum_xx + sum_yy - sum_xy
    
    def _sample(self, batch_size, dist='dirichlet', dirichlet_alpha=0.1, ori_data=None):
        if dist == 'dirichlet':
            z_true = np.random.dirichlet(
                np.ones(self.z_dim) * dirichlet_alpha, size=batch_size)
            z_true = torch.from_numpy(z_true).float()
            return z_true.to(self.device)
        elif dist == 'gaussian':
            z_true = np.random.randn(batch_size, self.z_dim)
            z_true = torch.softmax(torch.from_numpy(z_true), dim=1).float()
            return z_true.to(self.device)
        elif dist == 'gmm_std':
            odes = np.eye(self.z_dim) * 20
            ides = np.random.randint(low=0, high=self.z_dim, size=batch_size)
            mus = odes[ides]
            sigmas = np.ones((batch_size, self.z_dim)) * 0.2 * 20
            z_true = np.random.normal(mus, sigmas)
            z_true = F.softmax(torch.from_numpy(z_true).float(), dim=1)
            return z_true.to(self.device)
        elif dist == 'gmm_ctm' and ori_data is not None:
            with torch.no_grad():
                hid_vecs = self.encode(ori_data).cpu().numpy()
                gmm = GaussianMixture(n_components=self.z_dim, covariance_type='full', max_iter=200)
                gmm.fit(hid_vecs)
                # hid_vecs = torch.from_numpy(hid_vecs).to(self.device)
                gmm_spls, _spl_lbls = gmm.sample(n_samples=len(ori_data))
                theta_prior = torch.from_numpy(gmm_spls).float()
                theta_prior = F.softmax(theta_prior, dim=1)
                return theta_prior.to(self.device)
        else:
            return self.sample(dist='dirichlet', batch_size=batch_size)
       


         