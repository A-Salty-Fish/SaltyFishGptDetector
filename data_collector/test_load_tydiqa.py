from datasets import load_dataset, DownloadConfig

if __name__ == '__main__':
    dataset = load_dataset("wiki_qa")
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    print(len(train_dataset))
    print(train_dataset[0]['question'])
    print(train_dataset[0]['answer'])
    print(len(test_dataset))
    # for x in train_dataset:
    #     print(x)
