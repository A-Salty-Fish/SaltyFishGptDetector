# This is a sample Python script.
import json
import os


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def find_unique_words(text1, text2):
    # words1 = set(text1.split())
    words2 = set(text2.split())

    # unique_words = words1 - words2
    unique_words = []
    for t in text1.split():
        if t not in words2:
            unique_words.append(t)
    return unique_words



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # with open('./roberta_test/data/hc3_row.train', 'r', encoding='utf-8') as f:
    #     print(len(json.load(f)))

    # cheat_files = [
    #     '../data_collector/test_data/CHEAT/ieee-init.jsonl',
    #     '../data_collector/test_data/CHEAT/ieee-chatgpt-fusion.jsonl',
    #     '../data_collector/test_data/CHEAT/ieee-chatgpt-generation.jsonl',
    #     '../data_collector/test_data/CHEAT/ieee-chatgpt-polish.jsonl',
    # ]
    # for cheat_file in cheat_files:
    #     with open(cheat_file, 'r', encoding='utf-8') as f:
    #         jsons = []
    #         for line in f:
    #             jsons.append(json.loads(line))
    #         print(len(jsons))

    # ghostbuster_files = [
    #     '../data_collector/test_data/ghostbuster/essay_claude.txt',
    #     '../data_collector/test_data/ghostbuster/essay_gpt.txt',
    #     '../data_collector/test_data/ghostbuster/essay_gpt_semantic.txt',
    #     '../data_collector/test_data/ghostbuster/essay_gpt_writing.txt',
    #     '../data_collector/test_data/ghostbuster/essay_human.txt',
    # ]
    # for ghostbuster_file in ghostbuster_files:
    #     with open(ghostbuster_file, 'r', encoding='utf-8') as f:
    #             texts = []
    #             for line in f:
    #                 texts.append(line)
    #             print(len(texts))

    # m4_basedir = '../data_collector/test_data/m4/'
    # for file in os.listdir(m4_basedir):
    #     with open(m4_basedir + file, 'r', encoding='utf-8') as f:
    #         try:
    #             total = 0
    #             for line in f:
    #                 total+=1
    #             print(file + '\t' + str(total))
    #         except Exception as e:
    #             print(e)
    #             pass

    # base_dir = './dpo_test/qwen/'
    # for file in os.listdir(base_dir):
    #     if file.endswith('.test'):
    #         with open(base_dir + file, 'r', encoding='utf-8') as f:
    #             json_arr = json.load(f)
    #             ai_objs = [x for x in json_arr if x['label'] == 1]
    #             human_objs = [x for x in json_arr if x['label'] == 0]
    #             print(f'{file}: {len(human_objs)} : {len(ai_objs)}')
    #
    # count = 0
    # for file in os.listdir('D:\\Pupeteer_product_hunter\\data\\category_products'):
    #     with open('D:\\Pupeteer_product_hunter\\data\\category_products\\' + file, 'r' , encoding='utf-8') as f:
    #         if len(json.load(f)) == 0:
    #             print("\"https://www.producthunt.com/categories/" + file.replace('.json', '') + "\"")
    # print(count)

    # base_dir = './moe_test/data/'
    # for file in os.listdir(base_dir):
    #     count = 0
    #     with open(base_dir + file) as in_f:
    #         for line in in_f:
    #             count += 1
    #     print(file + ':' + str(count))

    # base_dir = 'D:\\毕设\\数据\\arxiv_pdf\\'
    # a = 0
    # b = 0
    # c = 0
    # d = 0
    # for file in os.listdir(base_dir):
    #     if file.startswith('07'):
    #         a+=1
    #     if file.startswith('08'):
    #         b+=1
    #     if file.startswith('09'):
    #         c+=1
    #     if file.startswith('10'):
    #         d+=1
    # print(a)
    # print(b)
    # print(c)
    # print(d)

    # count = 0
    # for file in os.listdir('D:\\Pupeteer_product_hunter\\data\\category_products\\'):
    #     with open('D:\\Pupeteer_product_hunter\\data\\category_products\\' + file, 'r', encoding='utf-8') as in_f:
    #         count += len(json.load(in_f))
    # print(count)
    #
    # count = 0
    # for file in os.listdir('D:\\Pupeteer_product_hunter\\data\\category_products_info\\'):
    #     with open('D:\\Pupeteer_product_hunter\\data\\category_products_info\\' + file, 'r', encoding='utf-8') as in_f:
    #         for obj in json.load(in_f):
    #             try:
    #                 if obj['name'] is not None and obj['description'] is not None and obj['url'] is not None:
    #                     count+=1
    #             except Exception as e:
    #                 pass
    # print(count)

    # with open('D:\\Pupeteer_product_hunter\\smallNameMap.json', 'r', encoding='utf-8') as in_f:
    #     arr = json.load(in_f)
    #     print(len(arr[0]))
    #     print(len(arr[1]))


    # text1 = '''Well, let me tell you, it was quite a tragic event for both Henry II and his opponent Gabriel de Montgomery.  It all went down in 1559, during a jousting match at the Hotel des Tournelles in Paris.  Henry was an experienced jouster, but Montgomery was a newcomer to the sport, and unfortunately, he was no match for the King's lance. During one of their runs, Henry's lance struck Montgomery's helmet, shattering it and sending a jagged piece of wood into his eye and brain.  The young man was rushed to a nearby hospital, but it was too late.  He died just a few days later, leaving behind a wife and children. As for Henry, he was devastated by the accident and reportedly went into a deep depression.  He blamed himself for Montgomery's death and was haunted by guilt for the rest of his life.  He even imposed a penance on himself, vowing to fast and do charitable works for the rest of his days. The incident also led to changes in the sport of jousting.  After Montgomery's death, many rules were put in place to make the sport safer, such as requiring jousters to wear full face and head protection.  Overall, it was a tragic event that had a lasting impact on the sport and on the King who inadvertently caused the death of his opponent.'''
    # text2 = '''Edmund Franks, a German jousting knight, was practicing his moves in the hall of the castle where he was staying when he accidentally killed Henry II's jousting opponent. Franks was immediately arrested and tried for murder, although many believed he was only acting in self-defense. Franks was found guilty and beheaded. He was 49 years old. + Henry II's jousting opponent was accidentally killed in 1559 by Edmund Franks, a German jousting knight. Franks was immediately arrested and tried for murder, although many believed he was only acting in self-defense. Franks was found guilty and beheaded. He was 49 years old. spects Edmund Franks - German jousting knight - was practicing his moves in the hall of the castle - where he was staying - when he accidentally killed Henry II's jousting opponent. Franks was immediately arrested and tried for murder, although many believed he was only acting in self-defense. Franks was found guilty and beheaded. He was 49 years old.'''
    #
    # print(find_unique_words(text1, text2))

    # text = "Nobelium (102No) is a synthetic element, and thus a standard atomic weight cannot be given. Like all synthetic elements, it has no stable isotopes. The first isotope to be synthesized (and correctly identified) was 254No in 1966. There are 13 known radioisotopes, which are 249No to 260No and 262No, and 4 isomers, 250mNo, 251mNo, 253mNo, and 254mNo. The longest-lived isotope is 259No with a half-life of 58 minutes. The longest-lived isomer is 251mNo with a half-life of 1.7 seconds.\n\nList of isotopes \n\n|-\n| rowspan=2|249No\n| rowspan=2 style=\"text-align:right\" | 102\n| rowspan=2 style=\"text-align:right\" | 147\n| rowspan=2|249.0878(3)#\n| rowspan=2|43.8(3.7) ms\n| \u03b1 \n| 245Fm\n| rowspan=2|5/2+#\n|-\n| SF?\n| (various)\n|-\n| rowspan=3|250No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 148\n| rowspan=3|250.08756(22)#\n| rowspan=3|5.7(8)\u00a0\u03bcs\n| SF (99.95%)\n| (various)\n| rowspan=3|0+\n|-\n| \u03b1(.05%)\n| 246Fm\n|-\n| \u03b2+ (2.5\u00d710\u22124%)\n| 250Md\n|-\n| style=\"text-indent:1em\" | 250mNo\n| colspan=\"3\" style=\"text-indent:2em\" | \n| 36(3)\u00a0\u03bcs\n| SF\n| (various)\n| \n|-\n| rowspan=3|251No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 149\n| rowspan=3|251.08894(12)#\n| rowspan=3|0.78(2)\u00a0s\n| \u03b1 (89%)\n| 247Fm\n| rowspan=3|7/2+#\n|-\n| SF (10%)\n| (various)\n|-\n| \u03b2+ (1%)\n| 251Md\n|-\n| style=\"text-indent:1em\" | 251mNo\n| colspan=\"3\" style=\"text-indent:2em\" | 110(180)#\u00a0keV\n| 1.7(10)\u00a0s\n|\n|\n| 9/2\u2212#\n|-\n| rowspan=3|252No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 150\n| rowspan=3|252.088967(10)\n| rowspan=3|2.27(14)\u00a0s\n| \u03b1 (73.09%)\n| 248Fm\n| rowspan=3|0+\n|-\n| SF (26.9%)\n| (various)\n|-\n| \u03b2+ (1%)\n| 252Md\n\n|-\n| rowspan=3|253No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 151\n| rowspan=3|253.090564(7)\n| rowspan=3|1.62(15)\u00a0min\n| \u03b1 (80%)\n| 249Fm\n| rowspan=3|(9/2\u2212)#\n|-\n| \u03b2+ (20%)\n| 253Md\n|-\n| SF (10\u22123%)\n| (various)\n|-\n| style=\"text-indent:1em\" | 253mNo\n| colspan=\"3\" style=\"text-indent:2em\" | 129(19)\u00a0keV\n| 31\u00a0\u03bcs\n|\n|\n| 5/2+#\n|-\n| rowspan=3|254No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 152\n| rowspan=3|254.090956(11)\n| rowspan=3|51(10)\u00a0s\n| \u03b1 (89.3%)\n| 250Fm\n| rowspan=3|0+\n|-\n| \u03b2+ (10%)\n| 254Md\n|-\n| SF (.31%)\n| (various)\n|-\n| rowspan=2 style=\"text-indent:1em\" | 254mNo\n| rowspan=2 colspan=\"3\" style=\"text-indent:2em\" | 500(100)#\u00a0keV\n| rowspan=2|0.28(4)\u00a0s\n| IT (80%)\n| 254No\n| rowspan=2|0+\n|-\n| \u03b1 (20%)\n| 250Fm\n|-\n| rowspan=2|255No\n| rowspan=2 style=\"text-align:right\" | 102\n| rowspan=2 style=\"text-align:right\" | 153\n| rowspan=2|255.093191(16)\n| rowspan=2|3.1(2)\u00a0min\n| \u03b1 (61.4%)\n| 251Fm\n| rowspan=2|(1/2+)\n|-\n| \u03b2+ (38.6%)\n| 255Md\n|-\n| rowspan=3|256No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 154\n| rowspan=3|256.094283(8)\n| rowspan=3|2.91(5)\u00a0s\n| \u03b1 (99.44%)\n| 252Fm\n| rowspan=3|0+\n|-\n| SF (.55%)\n| (various)\n|-\n| EC (.01%)\n| 256Md\n|-\n| rowspan=2|257No\n| rowspan=2 style=\"text-align:right\" | 102\n| rowspan=2 style=\"text-align:right\" | 155\n| rowspan=2|257.096888(7)\n| rowspan=2|25(2)\u00a0s\n| \u03b1 (99%)\n| 253Fm\n| rowspan=2|(7/2+)\n|-\n| \u03b2+ (1%)\n| 257Md\n|-\n| rowspan=3|258No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 156\n| rowspan=3|258.09821(11)#\n| rowspan=3|1.2(2)\u00a0ms\n| SF (99.99%)\n| (various)\n| rowspan=3|0+\n|-\n| \u03b1 (.01%)\n| 254Fm\n|-\n| \u03b2+\u03b2+ (rare)\n| 258Fm\n|-\n| rowspan=3|259No\n| rowspan=3 style=\"text-align:right\" | 102\n| rowspan=3 style=\"text-align:right\" | 157\n| rowspan=3|259.10103(11)#\n| rowspan=3|58(5)\u00a0min\n| \u03b1 (75%)\n| 255Fm\n| rowspan=3|(9/2+)#\n|-\n| EC (25%)\n| 259Md\n|-\n| SF (<10%)\n| (various)\n|-\n| 260No\n| style=\"text-align:right\" | 102\n| style=\"text-align:right\" | 158\n| 260.10264(22)#\n| 106(8)\u00a0ms\n| SF\n| (various)\n| 0+\n\n|-\n| 262No\n| style=\"text-align:right\" | 102\n| style=\"text-align:right\" | 160\n| 262.10746(39)#\n| ~5\u00a0ms\n| SF\n| (various)\n| 0+\n\nNucleosynthesis\n\nBy cold fusion\n208Pb(48Ca,xn)256\u2212xNo (x=1,2,3,4)\nThis cold fusion reaction was first studied in 1979 at the Flerov Laboratory of Nuclear Reactions (FLNR). Further work in 1988 at the GSI measured EC and SF branchings in 254No. In 1989, the FLNR used the reaction to measure SF decay characteristics for the two isomers of 254No. The measurement of the 2n excitation function was reported in 2001 by Yuri Oganessian at the FLNR.\n\nPatin et al. at the LBNL reported in 2002 the synthesis of 255\u2013251No in the 1-4n exit channels and measured further decay data for these isotopes.\n\nThe reaction has recently been used at the Jyvaskylan Yliopisto Fysiikan Laitos (JYFL) using the RITU set-up to study K-isomerism in 254No. The scientists were able to measure two K-isomers with half-lives of 275\u00a0ms and 198\u00a0s, respectively. They were assigned to 8\u2212 and 16+ K-isomeric levels.\n\nThe reaction was used in 2004\u20135 at the FLNR to study the spectroscopy of 255\u2013253No. The team were able to confirm an isomeric level in 253No with a half-life of 43.5\u00a0s.\n\n208Pb(44Ca,xn)252\u2212xNo (x=2)\nThis reaction was studied in 2003 at the FLNR in a study of the spectroscopy of 250No.\n\n207Pb(48Ca,xn)255\u2212xNo (x=2)\nThe measurement of the 2n excitation function for this reaction was reported in 2001 by Yuri Oganessian and co-workers at the FLNR. The reaction was used in 2004\u20135 to study the spectroscopy of 253No.\n\n206Pb(48Ca,xn)254\u2212xNo (x=1,2,3,4)\nThe measurement of the 1-4n excitation functions for this reaction were reported in 2001 by Yuri Oganessian and co-workers at the FLNR.\nThe 2n channel was further studied by the GSI to provide a spectroscopic determination of K-isomerism in 252No. A K-isomer with spin and parity 8\u2212 was detected with a half-life of 110\u00a0ms.\n\n204Pb(48Ca,xn)252\u2212xNo (x=2,3)\nThe measurement of the 2n excitation function for this reaction was reported in 2001 by Yuri Oganessian at the FLNR. They reported a new isotope 250No with a half-life of 36\u00a0\u03bcs. The reaction was used in 2003 to study the spectroscopy of 250No.They were able to observe two spontaneous fission activities with half-lives of 5.6\u00a0\u03bcs and 54\u00a0\u03bcs and assigned to 250No and 249No, respectively.\nThe latter activity was later assigned to a K-isomer in 250No. The reaction was reported in 2006 by Peterson et al. at the Argonne National Laboratory (ANL) in a study of SF in 250No. They detected two activities with half-lives of 3.7\u00a0\u00a0\u03bcs and 43\u00a0\u00a0\u03bcs and both assigned to 250No, the latter associated with a K-isomer. In 2020, a team at FLNR repeated this reaction and found a new 9.1-MeV alpha particle activity correlated to 245Fm and 241Cf, which they assigned to the new isotope 249No.\n\nBy hot fusion\n232Th(26Mg,xn)258\u2212xNo (x=4,5,6)\nThe cross sections for the 4-6n exit channels have been measured for this reaction at the FLNR.\n\n238U(22Ne,xn)260\u2212xNo (x=4,5,6)\nThis reaction was first studied in 1964 at the FLNR. The team were able to detect decays from 252Fm and 250Fm. The 252Fm activity was associated with an ~8\u00a0s half-life and assigned to 256102 from the 4n channel, with a yield of 45\u00a0nb.  \nThey were also able to detect a 10\u00a0s spontaneous fission activity also tentatively assigned to 256102.\nFurther work in 1966 on the reaction examined the detection of 250Fm decay using chemical separation and a parent activity with a half-life of ~50\u00a0s was reported and correctly assigned to 254102. They also detected a 10\u00a0s spontaneous fission activity tentatively assigned to 256102.\nThe reaction was used in 1969 to study some initial chemistry of nobelium at the FLNR. They determined eka-ytterbium properties, consistent with nobelium as the heavier homologue. In 1970, they were able to study the SF properties of 256No.\nIn 2002, Patin et al. reported the synthesis of 256No from the 4n channel but were unable to detect 257No.\n\nThe cross section values for the 4-6n channels have also been studied at the FLNR.\n\n238U(20Ne,xn)258\u2212xNo\nThis reaction was studied in 1964 at the FLNR. No spontaneous fission activities were observed.\n\n236U(22Ne,xn)258\u2212xNo (x=4,5,6)\nThe cross sections for the 4-6n exit channels have been measured for this reaction at the FLNR.\n\n235U(22Ne,xn)257\u2212xNo (x=5)\nThis reaction was studied in 1970 at the FLNR. It was used to study the SF decay properties of 252No.\n\n233U(22Ne,xn)255\u2212xNo\nThe synthesis of neutron deficient nobelium isotopes was studied in 1975 at the FLNR. In their experiments they observed a 250\u00a0s SF activity, which they tentatively assigned to 250No in the 5n exit channel. Later results have not been able to confirm this activity and it is currently unidentified.\n\n242Pu(18O,xn)260\u2212xNo (x=4?)\nThis reaction was studied in 1966 at the FLNR. The team identified an 8.2\u00a0s SF activity tentatively assigned to 256102.\n\n241Pu(16O,xn)257\u2212xNo\nThis reaction was first studied in 1958 at the FLNR. The team measured ~8.8\u00a0MeV alpha particles with a half-life of 30\u00a0s and assigned to 253,252,251102. A repeat in 1960 produced 8.9\u00a0MeV alpha particles with a half-life of 2\u201340\u00a0s and assigned to 253102 from the 4n channel. Confidence in these results was later diminished.\n\n239Pu(18O,xn)257\u2212xNo (x=5)\nThis reaction was studied in 1970 at the FLNR in an effort to study the SF decay properties of 252No.\n\n239Pu(16O,xn)255\u2212xNo\nThis reaction was first studied in 1958 at the FLNR. The team were able to measure ~8.8\u00a0MeV alpha particles with a half-life of 30\u00a0s and assigned to253,252,251102. A repeat in 1960 was unsuccessful and it was concluded the first results were probably associated with background effects.\n\n243Am(15N,xn)258\u2212xNo (x=4)\nThis reaction was studied in 1966 at the FLNR. The team were able to detect 250Fm using chemical techniques and determined an associated half-life significantly higher than the reported 3\u00a0s by Berkeley for the supposed parent 254No. Further work later the same year measured 8.1\u00a0MeV alpha particles with a half-life of 30\u201340\u00a0s.\n\n243Am(14N,xn)257\u2212xNo\nThis reaction was studied in 1966 at the FLNR. They were unable to detect the 8.1\u00a0MeV alpha particles detected when using a N-15 beam.\n\n241Am(15N,xn)256\u2212xNo (x=4)\nThe decay properties of 252No were examined in 1977 at Oak Ridge. The team calculated a half-life of 2.3\u00a0s and measured a 27% SF branching.\n\n248Cm(18O,\u03b1xn)262\u2212xNo (x=3)\nThe synthesis of the new isotope 259No was reported in 1973 from the LBNL using this reaction.\n\n248Cm(13C,xn)261\u2212xNo (x=3?,4,5)\nThis reaction was first studied in 1967 at the LBNL. The new isotopes 258No,257No and 256No were detected in the 3-5n channels. The reaction was repeated in 1970 to provide further decay data for 257No.\n\n248Cm(12C,xn)260\u2212xNo (4,5?)\nThis reaction was studied in 1967 at the LBNL in their seminal study of nobelium isotopes. The reaction was used in 1990 at the LBNL to study the SF of256No.\n\n246Cm(13C,xn)259\u2212xNo (4?,5?)\nThis reaction was studied in 1967 at the LBNL in their seminal study of nobelium isotopes.\n\n246Cm(12C,xn)258\u2212xNo (4,5)\nThis reaction was studied in 1958 by scientists at the LBNL using a 5% 246Cm curium target. They were able to measure 7.43\u00a0MeV decays from250Fm, associated with a 3\u00a0s 254No parent activity, resulting from the 4n channel. The 3\u00a0s activity was later reassigned to 252No, resulting from reaction with the predominant 244Cm component in the target. It could however not be proved that it was not due to the contaminant250mFm, unknown at the time.\nLater work in 1959 produced 8.3\u00a0MeV alpha particles with a half-life of 3\u00a0s and a 30% SF branch. This was initially assigned to 254No and later reassigned to 252No, resulting from reaction with the 244Cm component in the target.\nThe reaction was restudied in 1967 and activities assigned to 254No and 253No were detected.\n\n244Cm(13C,xn)257\u2212xNo (x=4)\nThis reaction was first studied in 1957 at the Nobel Institute in Stockholm. The scientists detected 8.5\u00a0MeV alpha particles with a half-life of 10 minutes. The activity was assigned to 251No or 253No. The results were later dismissed as background.\nThe reaction was repeated by scientists at the LBNL in 1958 but they were unable to confirm the 8.5\u00a0MeV alpha particles. \nThe reaction was further studied in 1967 at the LBNL and an activity assigned to 253No was measured.\n\n244Cm(12C,xn)256\u2212xNo (x=4,5)\nThis reaction was studied in 1958 by scientists at the LBNL using a 95% 244Cm curium target. They were able to measure 7.43\u00a0MeV decays from250Fm, associated with a 3\u00a0s 254No parent activity, resulting from the reaction (246Cm,4n). The 3\u00a0s activity was later reassigned to252No, resulting from reaction (244Cm,4n). It could however not be proved that it was not due to the contaminant 250mFm, unknown at the time.\nLater work in 1959 produced 8.3\u00a0MeV alpha particles with a half-life of 3\u00a0s and a 30% SF branch. This was initially assigned to 254No and later reassigned to 252No, resulting from reaction with the 244Cm component in the target.\nThe reaction was restudied in 1967 at the LBNL and a new activity assigned to 251No was measured.\n\n252Cf(12C,\u03b1xn)260\u2212xNo (x=3?)\nThis reaction was studied at the LBNL in 1961 as part of their search for element 104. They detected 8.2\u00a0MeV alpha particles with a half-life of 15\u00a0s. This activity was assigned to a Z=102 isotope. Later work suggests an assignment to 257No, resulting most likely from the \u03b13n channel with the 252Cf component of the californium target.\n\n252Cf(11B,pxn)262\u2212xNo (x=5?)\nThis reaction was studied at the LBNL in 1961 as part of their search for element 103. They detected 8.2\u00a0MeV alpha particles with a half-life of 15\u00a0s. This activity was assigned to a Z=102 isotope. Later work suggests an assignment to 257No, resulting most likely from the p5n channel with the 252Cf component of the californium target.\n\n249Cf(12C,\u03b1xn)257\u2212xNo (x=2)\nThis reaction was first studied in 1970 at the LBNL in a study of 255No. It was studied in 1971 at the Oak Ridge Laboratory. They were able to measure coincident Z=100 K X-rays from 255No, confirming the discovery of the element.\n\nAs decay products\nIsotopes of nobelium have also been identified in the decay of heavier elements. Observations to date are summarised in the table below:\n\nIsotopes\nTwelve radioisotopes of nobelium have been characterized, with the most stable being 259No with a half-life of 58 minutes. Longer half-lives are expected for the as-yet-unknown 261No and 263No. An isomeric level has been found in 253No and K-isomers have been found in 250No, 252No and 254No to date.\n\nChronology of isotope discovery\n\nNuclear isomerism\n254No\nThe study of K-isomerism was recently studied by physicists at the University of Jyv\u00e4skyl\u00e4 physics laboratory (JYFL). They were able to confirm a previously reported K-isomer and detected a second K-isomer. They assigned spins and parities of 8\u2212 and 16+ to the two K-isomers.\n\n253No\nIn 1971, Bemis et al. was able to determine an isomeric level decaying with a half-life of 31\u00a0s from the decay of 257Rf. This was confirmed in 2003 at the GSI by also studying the decay of 257Rf. Further support in the same year from the FLNR appeared with a slightly higher half-life of 43.5\u00a0s, decaying by M2 gamma emission to the ground state.\n\n252No\nIn a recent study by the GSI into K-isomerism in even-even isotopes, a K-isomer with a half-life of 110\u00a0ms was detected for 252No. A spin and parity of 8\u2212 was assigned to the isomer.\n\n250No\nIn 2003, scientists at the FLNR reported that they had been able to synthesise 249No, which decayed by SF with a half-life of 54\u00a0\u03bcs. Further work in 2006 by scientists at the ANL showed that the activity was actually due to a K-isomer in 250No. The ground state isomer was also detected with a very short half-life of 3.7\u00a0\u03bcs.\n\nChemical yields of isotopes\n\nCold fusion\nThe table below provides cross-sections and excitation energies for cold fusion reactions producing nobelium isotopes directly. Data in bold represents maxima derived from excitation function measurements. + represents an observed exit channel.\n\nHot fusion\nThe table below provides cross-sections and excitation energies for hot fusion reactions producing nobelium isotopes directly. Data in bold represents maxima derived from excitation function measurements. + represents an observed exit channel.\n\nRetracted isotopes\nIn 2003, scientists at the FLNR claimed to have discovered 249No, which would have been the lightest known isotope of nobelium. However, subsequent work showed that the 54\u00a0s activity was actually due to 250No and the isotope 249No was retracted. The discovery of this isotope was later reported in 2020; its decay properties differed from the 2003 claims.\n\nReferences \n\n Isotope masses from:\n\n Isotopic compositions and standard atomic masses from:\n\n Half-life, spin, and isomer data selected from the following sources.\n\n \nNobelium\nNobelium"
    # print(len(text.split(" ")))
    # content = text
    # # 截断过长的数据
    # words = content.split(' ')
    # if len(words) > 512:
    #     content = " ".join(words[0: 512])
    # print(content)

    # with open('./moe_test/data/nature/qwen/7.jsonl.qwen.rewrite.jsonl.train', 'r', encoding='utf-8') as in_f:
    #     json_objs = json.load(in_f)
    #     print(len([x for  x in json_objs if x['label'] == 0]))
    #     print(len([x for  x in json_objs if x['label'] == 1]))
    # with open('./moe_test/data/adversary/qwen/7.jsonl.qwen.rewrite.jsonl.qwen.paraphase.jsonl.train', 'r', encoding='utf-8') as in_f:
    #     json_objs = json.load(in_f)
    #     print(len([x for x in json_objs if x['label'] == 0]))
    #     print(len([x for x in json_objs if x['label'] == 1]))
    dirs = [
        './moe_test/data/adversary/qwen/',
        './moe_test/data/adversary/dpo/',
        './moe_test/data/adversary/dp/',
        './moe_test/data/nature/mix/',
        './moe_test/data/nature/qwen/',
        './moe_test/data/nature/glm/',
    ]

    for dir in dirs:
        for file in os.listdir(dir):
            if file.endswith('.test'):
                with open(dir + file, 'r' ,encoding='utf-8') as in_f:
                    json_objs = json.load(in_f)
                    print(file.split('/')[-1] + '\t' + str(len([x for x in json_objs if x['label'] == 0])) + '\t' + str(len([x for x in json_objs if x['label'] == 1])))

    pass
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
