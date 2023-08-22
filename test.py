import torch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from train import load_model, SoftIntroVAE, FantasySet


def test():
    pretrained = 'saves/fantasy_soft_intro_betas_1.0_8.0_0.5_model_epoch_1999_iter_27986.pth'
    device = torch.device('cuda')
    model = SoftIntroVAE(cdim=3, zdim=256, channels=[64, 128, 256, 512, 512, 512], image_size=256).to(device)
    load_model(model, pretrained, device)
    fantasy = FantasySet()
    z_space = []
    img_offset = []
    with torch.no_grad():
        for img in fantasy:
            x = img.to(device).unsqueeze(0)
            mu, logvar = model.encode(x)
            z = mu
            z_space.append(z)
            img_offset.append(OffsetImage(img.permute(1, 2, 0), zoom=0.3))

    z_space = torch.concatenate(z_space).cpu().numpy()

    z_embedded = TSNE(verbose=1).fit_transform(z_space)

    fig, ax = plt.subplots()

    x = z_embedded[:, 0]
    y = z_embedded[:, 1]

    ax.scatter(x, y)

    for x0, y0, img in zip(x, y, img_offset):
        ab = AnnotationBbox(img, (x0, y0), frameon=False)
        ax.add_artist(ab)

    fig.set_size_inches(20, 20)
    plt.show()


if __name__ == '__main__':
    test()