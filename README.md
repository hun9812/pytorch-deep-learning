# PyTorch 공부

텐서 연산부터 간단한 신경망 학습까지 연습한 코드입니다.

- [기초](01-pytorch-basics/): 텐서, 자동 미분, 선형·로지스틱·소프트맥스 회귀, `nn.Module`
- [MNIST](02-mnist-practice/01_mnist_classifier.py): 784 → 256 → 10 구조의 완전연결 신경망

## 실행

Python과 `torch`, `torchvision`이 필요합니다. 저장소 루트에서 실행합니다.

```bash
python 01-pytorch-basics/01_tensor_basics.py
python 02-mnist-practice/01_mnist_classifier.py
```

MNIST 데이터는 첫 실행 때 `data/`에 다운로드됩니다.
