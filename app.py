from flask import Flask, render_template, request, redirect, url_for, session
import torch
from app_model import BGM

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# can be changed according to 'vocab_size' used for training
vocab_size = 104

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BGM(vocab_size)
model.load_state_dict(torch.load('./b64_bl128_3e4_h8_l6_do0.2_10k.pth', map_location=device))
model.to(device)
model.eval()

with open('./war_and_peace.txt', 'r', encoding='utf-8') as f:
    text=f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode=lambda s : [stoi[c] for c in s]
decode= lambda s : ''.join([itos[i] for i in s])



@app.route('/', methods=['GET', 'POST'])
def index():
    ''' 
        index page which takes 'max_new_tokens' as input
    '''
    if request.method == 'POST':

        max_new_tokens = int(request.form['max_new_tokens'])

        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated_text = generate_text(model, context, max_new_tokens)
        session['generated_text'] = generated_text
        session['max_new_tokens'] = max_new_tokens

        return redirect(url_for('result'))

    return render_template('index.html')

@app.route('/result')
def result():
    ''' 
        result page which displays generated text.
    '''
    generated_text = session.pop('generated_text', '')
    max_new_tokens = session.pop('max_new_tokens', 0)

    return render_template('result.html', generated_text=generated_text, max_new_tokens=max_new_tokens)

def generate_text(model, context, max_new_tokens):

    generated_tokens = model.generate(context, max_new_tokens)[0].tolist()
    generated_text = ''.join([itos[i] for i in generated_tokens])
    return generated_text

# run app
if __name__ == '__main__':
    app.run(debug=True, port=8001)
