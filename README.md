# SKU110K_to_VOC
[SKU110K](https://github.com/eg4000/SKU110K_CVPR19)의 매대의 goods image를 custom goods image로 교체해 VOC annotation 생성

~~~
python3 SKU110K_sampler.py
~~~
or
~~~
# dir: SKU110K dataset 경로, num_files: 생성할 이미지 수(default=100), skip_p: 전체 annotation중 생략 비율(default=100), val_p: validation data 비율
sampler = SKU110KSampler(dir='./SKU110K_fixed/', num_files=10000, skip_p=0.0, val_p=0.1)
# dir: 배경 투명화된 상품이미지들(.png) 경로
sampler.generate_img_dataset(dir='images/')
sampler.generate_annotations()
~~~
