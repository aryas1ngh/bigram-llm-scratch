import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# hyperparameters
batch_size = 64
block_size = 128
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_head = 8
n_embd = batch_size * n_head
LAYERS = 6
dropout = 0.2

# change directory to './data/war_and_peace.txt' or whatever required.
with open('/kaggle/input/bigram/war_and_peace.txt', 'r', encoding='utf-8') as f:
    text=f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode=lambda s : [stoi[c] for c in s]
decode= lambda s : ''.join([itos[i] for i in s])

# data loading and splitting
data=torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_d=data[:n]
val_d=data[n:]


def gen_batch(split):
    
    data=train_d if split=='train' else val_d
    ix = torch.randint( len(data)-block_size,
                        (batch_size, ))                     # choose {batch_size} #integers from 0 to data.len - block_s
    
    x=torch.stack([data[i: i+block_size] for i in ix])
    y=torch.stack([data[i+1: i+block_size+1] for i in ix])  
    xb,yb=x.to(device),y.to(device)

    return xb,yb

@torch.no_grad()
def estimate_loss():
    '''
    Trains the model.
    Returns a dict of mean train & val split losses.
    '''
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = gen_batch(split)
            X, Y = X.to(device), Y.to(device)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



class Head (nn.Module) :
    ''' 
    single head of MHA.
    '''
    
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)        
        self.value = nn.Linear(n_embd, head_size, bias=False)        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))   # dont register as param
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        B,T,C = x.shape
        
        # attention(q,k,v) = softmax(q.kT/sqrt(d_k))*v
        
        k = self.key(x)
        q = self.query (x)
        wt = q @ k.transpose(-2, -1) * C ** -0.5
        
        wt = wt.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        wt = F.softmax(wt, dim=-1)
        wt = self.dropout(wt)
        
        v = self.value(x)
        out = wt @ v
        return out


class MultiHeadedAttention (nn.Module) :
    
    '''
    multiheaded att. 
    -> calculate Head() for each head.
    -> linear layer pass.
    -> concatenate.
    -> dropout.
    '''
    
    def __init__(self, n_head, head_size):
        
        super().__init__()
        self.heads =nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj=nn.Linear(n_embd, n_embd)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        
        return out



class FeedForward (nn.Module) :
    '''
    feed forward layer.
    linear: n_embd --> [] [] [] []
    relu
    linear: [] [] [] [] --> n_embd
    dropout
    '''
    def __init__(self, n_embd) :
        super().__init__()
        self.net = nn.Sequential(
            
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)




class Block (nn.Module) :
    '''
    transformer block. 
    X = MHA( LayerNorm (X) ) + FF ( LayerNorm (X) )
    '''
    def __init__(self, n_embd, n_head) :
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadedAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x) :
        x = x + self.sa(self.ln1(x))
        x = x +  self.ff(self.ln2(x))
        return x

class BGM(nn.Module) :
    def __init__(self,vocab_size):
        '''
        token_emb_tab : Token Embedding Table with V embeddings of size V.
        pos_emb_tab : Positional embedding table.
        blocks : {LAYERS} number of blocks.
        ln_final : final ln layer
        lm_head : 
        '''
        super().__init__()
        self.token_emb_tab = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=n_embd)
        self.pos_emb_tab = nn.Embedding (block_size, n_embd)
        self.blocks = nn.Sequential ( *[Block(n_embd, n_head=n_head) for _ in range(LAYERS)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward( self, index, targets=None) :
        '''fpass : squeeze logits, targets then calc celoss'''
        
        B,T = index.shape
        tok_emb=self.token_emb_tab(index)
        pos_emb=self.pos_emb_tab(torch.arange(T, device=device))
        x = tok_emb + pos_emb # !!!!
        x = self.blocks(x)
        x = self.ln_final(x)
        
        logits = self.lm_head(x)
        
        
        if targets==None:
            loss=None
        else:
            b,t,c=logits.shape
            logits=logits.view(b*t, c)
            targets=targets.view(b*t)
            loss=F.cross_entropy(logits,targets)
            
        return logits,loss
    
    def generate( self, index, max_new_tokens):
        '''
            do fpass for index. take last logit. softmax. 
            calc new index by MN sampling. cat to previous index.
            repeat for max_token length.
        '''
        for _ in range(max_new_tokens):
            
            # crop idx to the last block_size tokens
            idx_cropd = index[:, -block_size:]
            
            logits, loss = self(idx_cropd) 
            logits = logits [:, -1, :] 
            p=F.softmax(logits, dim=-1)
            
            next_i = torch.multinomial(p, num_samples=1)
            index = torch.cat((index, next_i),dim=1)
            
        return index

m = BGM(vocab_size)
m = m.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M trainable parameters.')
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)


for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"Epoch {iter}: Training loss : {losses['train']:.4f}, Validation loss : {losses['val']:.4f}")

    xb, yb = gen_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1) , dtype=torch.long, device=device)

# test run the model, change max_new_tokens as required.
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

# save model according to directory
torch.save(m.state_dict(), '/kaggle/working/b64_bl128_3e4_h8_l6_do0.2_10k.pth')




