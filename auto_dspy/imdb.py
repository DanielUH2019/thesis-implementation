import tqdm
import random

from datasets import load_dataset
from dspy.datasets.dataset import Dataset

class Imdb:
    def __init__(self) -> None:
        super().__init__()
        self.do_shuffle = False

        dataset = load_dataset("imdb")

        hf_official_train = dataset['train']
        hf_official_test = dataset['test']
        official_train = []
        official_test = []

        for example in tqdm.tqdm(hf_official_train):
            text = example['text']

            answer = example['label']
            
            
            # gold_reasoning = ' '.join(answer[:-2])
            answer = parse_number_to_sentiment(answer)

            official_train.append(dict(text=text, answer=answer))

        for example in tqdm.tqdm(hf_official_test):
            text = example['text']

            answer = example['label']
            # assert answer[-2] == '####'
            
            # gold_reasoning = ' '.join(answer[:-2])
            answer = parse_number_to_sentiment(answer)

            official_test.append(dict(text=text, answer=answer))

        rng = random.Random(0)
        rng.shuffle(official_train)

        rng = random.Random(0)
        rng.shuffle(official_test)

        trainset = official_train[:30]
        devset = official_test[:25]
        testset = official_test[30:]

        import dspy

        trainset = [dspy.Example(**x).with_inputs('text') for x in trainset]
        devset = [dspy.Example(**x).with_inputs('text') for x in devset]
        testset = [dspy.Example(**x).with_inputs('text') for x in testset]

        # print(f"Trainset size: {len(trainset)}")
        # print(f"Devset size: {len(devset)}")
        # print(f"Testset size: {len(testset)}")

        self.train = trainset
        self.dev = devset
        self.test = testset



def imdb_metric(real, pred, trace=None):
    result = pred.answer.lower()
    if result == 'neg':
        result = 'negative'
    if result == 'pos':
        result = 'positive'
    return real.answer == result

def parse_number_to_sentiment(number: int):
    if number == 0:
        return 'negative'
    
    return 'positive'