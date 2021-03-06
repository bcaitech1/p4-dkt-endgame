# Stage 4 - DKT

> 2021 boostcamp AI Tech

## Task

### **๐ Knowledge Tracing๋?**

**Knowledge Training์ ์ฌ๋์ ์ง์ ์ํ๋ฅผ ์ถ์ ํ๋ ๋ฐฉ๋ฒ๋ก ์๋๋ค.**

- ์ํ์ ๋ณด๋ ๊ฒ์ ๋์ผํ์ง๋ง ๋จ์ํ ์ฐ๋ฆฌ๊ฐ ์ํ์ 80์  ๋ง์๋ค๊ณ  ์๋ ค์ฃผ๋ ๊ฒ์ ๋์ด์ ์ฐ๋ฆฌ๊ฐ ์ํ์ด๋ผ๋ ๊ณผ๋ชฉ์ ์ผ๋ง๋งํผ ์ดํดํ๊ณ  ์๋์ง ์ธก์ ํฉ๋๋ค. ์ถ๊ฐ์ ์ผ๋ก ์ด๋ฐ ์ดํด๋๋ฅผ ํ์ฉํ์ฌ ์ฐ๋ฆฌ๊ฐ ์์ง ํ์ง ์์ ๋ฏธ๋์ ๋ฌธ์ ์ ๋ํด ์ฐ๋ฆฌ๊ฐ ๋ง์์ง ํ๋ฆด์ง ์์ธก์ด ๊ฐ๋ฅํฉ๋๋ค!
- ๋ํ๋ **๋ฏธ๋์ ๋ฌธ์ ์ ๋ํด์ ๋ง์ถ์ง ํ๋ฆด์ง ์์ธก**ํ๋ ๊ฒ์ ์ง์ค๋์ด ์์ต๋๋ค.
- ๋ํ๋ฅผ ๋ฒ์ด๋ ์ ํฌ๋ ์ฃผ์ด์ง ๋ฌธ์ ๋ฅผ ๋งํ๋ ๋ฐ ์์ด ์ด๋ ํ ๊ฒฝํ๋ค, ์ฆ **ํ์์ ์ฑ์ฅ์ ์์ด ์ค์ํ ์์๊ฐ ๋ฌด์์ธ์ง**๋ฅผ ํ์ธํ๋ ๊ฒ์ ์ด์ ์ ๋ง์ถ์์ต๋๋ค.
- ๋ฐ๋ผ์ ์ ํฌ๋ ๊ธฐ์กด์ ์์ธก๋ง ํด์ฃผ๋ Knowledge Tracing ๋ชจ๋ธ์์ ๋ฒ์ด๋ **์น์ ํ** ๋ชจ๋ธ์ ๋ง๋ค๊ณ ์ ํ์ต๋๋ค. ์ ํฌ๋ **[Context-Aware Attentive Knowledge Tracing (AKT)](https://arxiv.org/abs/2007.12324)** ๋ชจ๋ธ์ ๊ธฐ๋ฐ์ผ๋ก [**ELO rating system**](https://www.fi.muni.cz/~xpelanek/publications/CAE-elo.pdf) ์ ์ ์ฉํ์ฌ ๋ฐ์ ์์ผฐ์ต๋๋ค.

# Model Architecture

based by [Context-Aware Attentive Knowledge Tracing (AKT)](https://arxiv.org/abs/2007.12324)

![pipeline2](https://user-images.githubusercontent.com/56197411/122345523-e8997000-cf82-11eb-968b-33c11b7b304d.PNG)

[AKT](https://github.com/arghosh/AKT)๋ Question Transformer Encoder์ Knowledge Transformer Encoder๋ฅผ ํตํด ๋ฌธ์ ์ ์ ํ์ ๋ํ ํ์ต, ์ ํ๊ณผ ์ฌ์ฉ์์ ์๋ต์ ๋ํ ํ์ต์  
๊ฐ๊ฐ ์งํํฉ๋๋ค. Encoder๋ฅผ ํตํด ์ฌ๊ตฌ์ฑํ Sequence๋ฅผ Monotonic Attention ๊ตฌ์กฐ๋ฅผ ํตํด ๋ค์ ๋ฌธ์ ์ ๋ํ ์๋ต์ ์์ธกํฉ๋๋ค.

## Monotonic Attention

![monotonic](https://user-images.githubusercontent.com/56197411/122346770-4aa6a500-cf84-11eb-95c5-56228be6759e.PNG)  
๊ฐ Transformer Layer์ ์ฌ์ฉํ๋ Monotonic Attention์ ํ์ฅ๋ Attention ๊ตฌ์กฐ์๋๋ค. ๋น์ทํ ์ ํ์ผ์๋ก, ์ต๊ทผ์ ๋ฐฐ์ด ์ ํ์ผ์๋ก ๋ ๊ฐํ๊ฒ  
์์ฉํฉ๋๋ค.

# Usage

## AKT_ELO

### Train & Inference

`shell $ python main.py `  
`shell $ python inference.py `

## ELO

### Train & Inference

`shell $ python elo.py `

## Other Models Available

- SAINT  
  ๋ณ๋ ํด๋์ ์ฝ๋์ ์ฌ์ฉ์์๊ฐ ์กด์ฌํฉ๋๋ค.

# Members

## Team ENDGAME

| ๊นํ๋ง๋ฃจ | ์์ฌ์ด |               ์ด๋ํ                | ์ ์งํ |                ์ต์ด์                 | ํ์น์ฐ |
| :------: | :----: | :---------------------------------: | :----: | :-----------------------------------: | :----: |
|  github  | github | [github](https://github.com/Hoon94) | github | [github](https://github.com/iseochoi) | github |
