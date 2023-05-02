import pickle
import pandas as pd
DATA_PATH = '.'



train_csv = DATA_PATH + '/train_sent_emo-v1.csv'
dev_csv = DATA_PATH + '/dev_sent_emo-v1.csv'
test_csv = DATA_PATH + '/test_sent_emo-v1.csv'


train_text = pd.read_csv(train_csv)
dev_text = pd.read_csv(dev_csv)
test_text = pd.read_csv(test_csv)

frames = [train_text, dev_text, test_text]
results = pd.concat(frames)

results.to_csv(DATA_PATH+'/all_sent_emo-v1.csv', index=0)

