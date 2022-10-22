@echo off
SetLocal

set S=(1, 1, 5)

set N=(lstm vdcnn)
for %%n in %N% do (
	for /L %%s in %S% do (
		python main.py --dataset news --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed %%s --model1 lstm --model2 %%n
	)
)

set N=(vdcnn)
for %%n in %N% do (
	for /L %%s in %S% do (
		python main.py --dataset news --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed %%s --model1 vdcnn --model2 %%n
	)
)

set N=(fcn cnn lstm vdcnn)
for %%n in %N% do (
	for /L %%s in %S% do (
		python main.py --dataset news --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed %%s --model1 fcn --model2 %%n
	)
)

set N=(cnn lstm vdcnn)
for %%n in %N% do (
	for /L %%s in %S% do (
		python main.py --dataset news --model_type coteaching_plus --noise_type symmetric --noise_rate 0.2 --seed %%s --model1 cnn --model2 %%n
	)
)

EndLocal