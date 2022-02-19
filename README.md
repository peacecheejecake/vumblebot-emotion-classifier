# vumblebot-emotion-classifier

## Usage
```
python run.py -i "여기에 입력이 들어갑니다"
```

인자 | 설명 | 기본값
--- | --- | ---
`-i`, `--input` | 입력 텍스트 | 
`-m`, `--model_path` | 모델 가중치 경로 | ./asset/tmp-albert.pth
`-a`, `--albert` | True면 Albert, False면 Bert | True
`--cpu` | CPU 사용. False이고 GPU가 있으면 cuda | False

## Dependencies
* transformers >= 4.15
