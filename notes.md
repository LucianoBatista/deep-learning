# Setup

Como vamos usar GPU, estarei utilizando o tensorflow pelo tensorman do POP_OS. O Pytorch **TODO!**

## Commands

Para rodar a imagem base original do tensorflow com suporte ao tensorflow, basta rodar os comandos abaixo.

```bash
tensorman run -p 8888:8888 --gpu --python3 --jupyter bash
jupyter notebook --ip=0.0.0.0 --no-browser
```

Também é possível criar uma imagem própria com libs que iremos utilizar durante os estudos, ver [documentação](https://support.system76.com/articles/tensorman).

Se quisermos instalar os toolkits da Nvidia pela system76, ver [doc](https://support.system76.com/articles/cuda/).

A instalação do Pytorch ocorreu tranquila no environment, and is already using the gpu, with cuda toolkit 10.2.


## Pytorch

Instalei o Pytorch no ambiente principal do python na máquina, gerido pelo pyenv. Caso seja necessário alterar a versão do python, podemos facilmente alternar isso com o pyenv e utilizar outros pacotes. Mas os comandos básicos do site do pytorch funciona perfeitamente para fornecer suporte a GPU usando o pytorch.