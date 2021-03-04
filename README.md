第七周：
作业要求：
Bert-based DST
• https://github.com/laituan245/BERT-Dialog-State-Tracking
• https://arxiv.org/pdf/1910.12995.pdf
• 根据以上论⽂和代码，在CrossWOZ数据集上训练DST模型

作业思路：
WOZ是英文数据集，CrossWOZ是中文数据集，因此在生成feature的时候用Jieba进行分词。另外CrossWOZ数据集的结构和WOZ数据集有差异，因此对数据处理的部分进行修改，主要修改了以下模块的以下内容：

dataset.py 
	class Turn
	class Dataset
	class Ontology

main.py 
	load_dataset()

训练脚本：
python main.py --do_train --data_dir=data/ --bert_model=bert-base-chinese --output_dir=outputs --epochs=1

训练效果：
时间仓促，只完成了代码部分，没有来得及进行模型训练，后面有时间进行模型训练之后再更新训练的效果