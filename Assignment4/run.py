"""Reproducible experiment entry point; paths are relative to this file."""
import argparse
import csv
import json
import platform
import random
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from src.data import load_train_val_datasets, load_test_dataset
from src.models import classifier, Denoiser, Diffusion

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


def checkpoint(obj, path):
    # Atomic replacement and retries tolerate short Windows indexing/sync locks.
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(obj, stream)
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.2)


def save(name, obj):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def grid(images, filename, titles=None, columns=10):
    images = images.detach().cpu().reshape(-1, 28, 28)
    rows = (len(images)+columns-1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns*1.25, rows*1.45), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis('off')
        if i < len(images):
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            if titles is not None:
                ax.set_title(titles[i], fontsize=7)
    fig.tight_layout(pad=.4)
    fig.savefig(OUT / filename, dpi=140)
    plt.close(fig)


@torch.no_grad()
def predict(model, x, batch=512):
    model.eval()
    return torch.cat([model(part) for part in x.split(batch)])


def datasets(cfg, device):
    train, val = load_train_val_datasets(ROOT / cfg['data_dir'], cfg['train_size'], cfg['split_seed'])
    test = load_test_dataset(ROOT / cfg['data_dir'])
    x, y = train.dataset.tensors
    result = [x[train.indices], y[train.indices], x[val.indices], y[val.indices], *test.tensors]
    save('split.json', {'seed': cfg['split_seed'], 'train_indices': train.indices, 'val_indices': val.indices})
    return [v.to(device) for v in result]


def architecture(model):
    x = torch.zeros(1, 1, 28, 28)
    rows = []
    for layer in model:
        x = layer(x)
        rows.append({'layer': str(layer), 'output': list(x.shape),
                     'parameters': sum(p.numel() for p in layer.parameters())})
    return rows


def classify(cfg, data, device):
    x, y, vx, vy, tx, ty = data
    permutation = torch.randperm(784, generator=torch.Generator().manual_seed(cfg['permutation_seed']))
    np.save(OUT/'permutation.npy', permutation.numpy())
    grid(torch.cat([x[:10], x[:10].flatten(1)[:, permutation].reshape(-1,1,28,28)]), 'permutation.png')
    results = []
    c = cfg['classification']
    for permuted in (False, True):
        inputs = [a.flatten(1)[:, permutation].reshape_as(a) if permuted else a for a in (x, vx, tx)]
        for name in ('mlp', 'cnn'):
            key = name + ('_permuted' if permuted else '_original')
            seed_all(cfg['seed'])
            model = classifier(name)
            save(key+'_architecture.json', architecture(model))
            model.to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=c['learning_rate'])
            generator = torch.Generator().manual_seed(cfg['seed'])
            history, best, best_epoch = [], -1, 0
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            sync(device)
            start = time.perf_counter()
            for epoch in range(1, c['epochs']+1):
                model.train()
                order = torch.randperm(len(x), generator=generator).to(device)
                total_loss, correct = 0., 0
                for ids in order.split(c['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs[0][ids])
                    loss = F.cross_entropy(logits, y[ids])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()*len(ids)
                    correct += (logits.argmax(1)==y[ids]).sum().item()
                logits = predict(model, inputs[1])
                accuracy = (logits.argmax(1)==vy).float().mean().item()
                history.append({'epoch': epoch, 'train_loss': total_loss/len(x), 'train_accuracy': correct/len(x),
                                'val_loss': F.cross_entropy(logits, vy).item(), 'val_accuracy': accuracy})
                if accuracy > best:
                    best, best_epoch = accuracy, epoch
                    checkpoint(model.state_dict(), OUT/(key+'.pt'))
                print(f'{key} epoch {epoch}: train={correct/len(x):.4f} val={accuracy:.4f}', flush=True)
                save(key+'_history.json', history)
            sync(device)
            elapsed = time.perf_counter()-start
            memory = torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
            model.load_state_dict(torch.load(OUT/(key+'.pt'), map_location=device, weights_only=True))
            sync(device)
            start = time.perf_counter()
            logits = predict(model, inputs[2])
            sync(device)
            test_seconds = time.perf_counter()-start
            predictions = logits.argmax(1)
            cm = torch.bincount(ty*10+predictions, minlength=100).reshape(10,10).cpu().numpy()
            confusion_pairs = [(int(cm[a,b]), NAMES[a], NAMES[b]) for a in range(10) for b in range(10) if a!=b]
            confusion_pairs.sort(reverse=True)
            result = {'name': key, 'parameters': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch,
                      'val_accuracy': best, 'test_accuracy': (predictions==ty).float().mean().item(),
                      'test_loss': F.cross_entropy(logits,ty).item(), 'train_seconds': elapsed,
                      'test_seconds': test_seconds, 'peak_allocated_MiB': memory,
                      'confusion_matrix': cm.tolist(), 'recall': (cm.diagonal()/cm.sum(1)).tolist(),
                      'top_errors': confusion_pairs[:5]}
            results.append(result)
            save('classification.json', results)
            wrong = (predictions!=ty).nonzero().flatten()[:20]
            grid(tx[wrong], key+'_errors.png', [f'{NAMES[a]} > {NAMES[b]}' for a,b in zip(ty[wrong].tolist(),predictions[wrong].tolist())])
    fields = ['name','parameters','best_epoch','val_accuracy','test_accuracy','train_seconds','test_seconds','peak_allocated_MiB']
    with (OUT/'classification.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(results)
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    for r in results:
        h = json.loads((OUT/(r['name']+'_history.json')).read_text())
        axes[0].plot([v['epoch'] for v in h], [v['val_accuracy'] for v in h], label=r['name'])
    axes[0].set(xlabel='Epoch',ylabel='Validation accuracy'); axes[0].legend(fontsize=7)
    axes[1].bar([r['name'].replace('_','\n') for r in results],[r['test_accuracy'] for r in results])
    axes[1].set(ylabel='Test accuracy',ylim=(0,1))
    fig.tight_layout(); fig.savefig(OUT/'comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ax,r in zip(axes.flat,results):
        ax.imshow(r['confusion_matrix'],cmap='Blues')
        ax.set(title=r['name'],xlabel='Predicted label',ylabel='True label',xticks=range(10),yticks=range(10))
    fig.tight_layout(); fig.savefig(OUT/'confusion.png',dpi=140); plt.close(fig)


def diffusion_train(cfg, data, device):
    seed_all(cfg['seed'])
    c = cfg['diffusion']
    x = data[0]*2-1
    model = Denoiser(c['channels']).to(device)
    process = Diffusion(c['steps'],device)
    optimizer = torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    generator = torch.Generator().manual_seed(cfg['seed'])
    history = []
    sync(device); start = time.perf_counter()
    for epoch in range(1,c['epochs']+1):
        model.train(); total=0.
        order = torch.randperm(len(x),generator=generator).to(device)
        for ids in order.split(c['batch_size']):
            clean = x[ids]
            t = torch.randint(c['steps'],(len(ids),),device=device)
            noise = torch.randn_like(clean)
            loss = F.mse_loss(model(process.add_noise(clean,t,noise),t),noise)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += loss.item()*len(ids)
        history.append({'epoch':epoch,'noise_mse':total/len(x)})
        print(f'diffusion epoch {epoch}: mse={total/len(x):.6f}',flush=True)
        save('diffusion_history.json',history)
        checkpoint(model.state_dict(),OUT/'diffusion.pt')
    sync(device)
    save('diffusion_training.json',{'samples':len(x),'input_channels':1,'base_channels':c['channels'],
         'steps':c['steps'],'epochs':c['epochs'],'seed':cfg['seed'],'device':str(device),
         'device_name':torch.cuda.get_device_name() if device.type=='cuda' else platform.processor(),
         'train_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),
         'terminal_alpha_bar':process.abar[-1].item(),'beta_start':0.0001,'beta_end':0.05})


@torch.no_grad()
def generate(cfg,data,device):
    c=cfg['diffusion']; model=Denoiser(c['channels']).to(device)
    model.load_state_dict(torch.load(OUT/'diffusion.pt',map_location=device,weights_only=True)); model.eval()
    process=Diffusion(c['steps'],device)
    generator=torch.Generator(device=device).manual_seed(c['sample_seed'])
    initial=torch.randn((c['samples'],1,28,28),generator=generator,device=device)
    torch.save(initial.cpu(),OUT/'initial_noise.pt')
    sync(device); start=time.perf_counter()
    samples,snapshots=process.sample(model,initial,c['sample_seed']+1)
    sync(device); seconds=time.perf_counter()-start
    samples=(samples+1)/2
    torch.save(samples.cpu(),OUT/'samples.pt')
    grid(samples,'samples.png')
    grid(torch.cat([(s+1)/2 for s in snapshots]),'reverse.png',columns=8)
    clean=data[0][:8]*2-1
    noise=torch.randn(clean.shape,generator=generator,device=device)
    forward=[(clean+1)/2]
    for t in [0,c['steps']//4,c['steps']//2,c['steps']-1]:
        forward.append((process.add_noise(clean,torch.full((8,),t,device=device),noise)+1)/2)
    grid(torch.cat(forward),'forward.png',columns=8)
    judge=classifier('cnn').to(device)
    judge.load_state_dict(torch.load(OUT/'cnn_original.pt',map_location=device,weights_only=True))
    probabilities=predict(judge,samples).softmax(1)
    confidence,labels=probabilities.max(1)
    counts=torch.bincount(labels,minlength=10).cpu().tolist()
    # Exact pixel-MSE nearest neighbors over ALL 50,000 training images, chunked.
    nearest_distance=torch.full((len(samples),),float('inf'),device=device)
    nearest_index=torch.zeros(len(samples),dtype=torch.long,device=device)
    flat=samples.flatten(1)
    for offset in range(0,len(data[0]),1000):
        ref=data[0][offset:offset+1000].flatten(1)
        distances=(flat.square().sum(1)[:,None]+ref.square().sum(1)[None]-2*flat@ref.T).clamp_min(0)/784
        values,indices=distances.min(1)
        update=values<nearest_distance
        nearest_distance[update]=values[update]; nearest_index[update]=indices[update]+offset
    pairs=torch.stack([samples[:10],data[0][nearest_index[:10]]],1).flatten(0,1)
    grid(pairs,'nearest.png',columns=10)
    failures=confidence.argsort()[:20]
    grid(samples[failures],'failures.png',[f'#{i}: {confidence[i]:.2f}' for i in failures.tolist()])
    save('generation.json',{'sample_seed':c['sample_seed'],'reverse_noise_seed':c['sample_seed']+1,
        'sample_count':len(samples),'sample_seconds':seconds,'seconds_per_image':seconds/len(samples),
        'predicted_class_counts':counts,'mean_confidence':confidence.mean().item(),
        'low_confidence_indices':failures.cpu().tolist(),'nearest_train_subset_indices':nearest_index.cpu().tolist(),
        'nearest_pixel_mse':nearest_distance.cpu().tolist(),
        'note':'Classifier predictions are a coverage proxy, not human labels. Low confidence is a failure candidate criterion, not proof of failure.'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].bar(range(10),counts); axes[0].set(xlabel='CNN predicted class',ylabel='Count / 100',xticks=range(10))
    h=json.loads((OUT/'diffusion_history.json').read_text())
    axes[1].plot([v['epoch'] for v in h],[v['noise_mse'] for v in h]); axes[1].set(xlabel='Epoch',ylabel='Noise MSE')
    fig.tight_layout(); fig.savefig(OUT/'generation_metrics.png',dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','classify','diffusion','sample'],default='all')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config/config.json').read_text())
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    device=torch.device(args.device)
    save('config_used.json',cfg)
    save('environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'platform':platform.platform(),'device':str(device),'gpu':torch.cuda.get_device_name() if device.type=='cuda' else None})
    data=datasets(cfg,device)
    if args.stage in ('all','classify'): classify(cfg,data,device)
    if args.stage in ('all','diffusion'): diffusion_train(cfg,data,device)
    if args.stage in ('all','sample'): generate(cfg,data,device)


if __name__=='__main__': main()
