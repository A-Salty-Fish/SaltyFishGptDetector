# Use a pipeline as a high-level helper
from transformers import pipeline

def init_classifier():
    pipe = pipeline("text-classification", model="roberta-large-openai-detector")

    return pipe


def classify_is_human(classfier, text, bar=0.50000):
    res_0 = classfier(text)[0]
    # print(res_0)
    if res_0['label'] == 'LABEL_0':
        return res_0['score'] >= bar
    else:
        return res_0['score'] < bar

json_1 = {"question":"Historical P\/E ratios of small-cap vs. large-cap stocks?","human_answers":["There is most likely an error in the WSJ's data.  Yahoo! Finance reports the P\/E on the Russell 2000 to be 15 as of 8\/31\/11 and S&P 500 P\/E to be 13 (about the same as WSJ). Good catch, though!  E-mail WSJ, perhaps they will be grateful."],"chatgpt_answers":["Historical price-to-earnings (P\/E) ratios for small-cap and large-cap stocks can vary significantly over time and may not be directly comparable due to the different characteristics of these two categories of stocks.Small-cap stocks, which are defined as stocks with a market capitalization of less than $2 billion, tend to be riskier and more volatile than large-cap stocks, which have a market capitalization of $10 billion or more. As a result, investors may be willing to pay a higher price for the potential growth opportunities offered by small-cap stocks, which can lead to higher P\/E ratios.On the other hand, large-cap stocks tend to be more established and stable, with a longer track record of earnings and revenue growth. As a result, these stocks may trade at lower P\/E ratios, as investors may be less willing to pay a premium for their growth potential.It is important to note that P\/E ratios are just one factor to consider when evaluating a stock and should not be used in isolation. Other factors, such as the company's financial health, industry trends, and macroeconomic conditions, can also impact a stock's P\/E ratio."]}

if __name__ == '__main__':
    classifier = init_classifier()
    print(classify_is_human(classifier, "zerogpt is an advanced and reliable chat GPT detector tool designed to analyze text and determine if it was generated by a human or an AI-powered language model. is zerogpt reliable ? "))
    print(classify_is_human(classifier, json_1['human_answers'][0]))
    print(classify_is_human(classifier, json_1['chatgpt_answers'][0]))