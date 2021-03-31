import pandas as pd

import config
import utils
import data_loader
import data_path




def extract_feature(text):
    features = {}

    normalized_text = normalize_text(text)

    # tagger = hazm.POSTagger(model=data_path.POSTAGGER_MODEL_PATH)
    # tags = tagger.tag(hazm.word_tokenize(normalized_text))
    # tags_list = [i[1] for i in tags]

    # alphabets_in_texts = [i for i in Counter(normalized_text) if i in config.alphabet]

    words, sentences = tokenize_word_and_sentence(normalized_text)

    features['BOW'] = words
    # C = len(normalized_text)
    # N = len(words)

    # # ویژگی های مبتنی بر نویسه
    # # تعداد کل نویسه ها = C
    # features['F1'] = C
    # # تعداد کل حروف الفبا / C
    # features['F2'] = len(alphabets_in_texts)/C
    # # # تعداد حروف الفبا
    # features['F49'] = len(alphabets_in_texts)
    # # تعداد کل اعداد
    # features['F3'] = tags_list.count('NUM')
    # # تعداد نویسه فاصله
    # features['F4'] = text.count(' ')/C
    # # تعداد نویسه تب
    # features['F5'] = text.count('\t')/C
    # # تعداد نویسه ویژه
    # features['F6'] = utils.count_chars(normalized_text , config.special_chars)/C

    # #ویژگی های مبتنی بر واژه 
    # # تعداد کل کلمات
    # features['F7'] = N
    # #میانگین تعداد نویسه در هر کلمه
    # features['F8'] = C/N
    # # غنای واژگانی (کل کلمات یکتا تقسیم بر تعداد کل کلمه ها)
    # V = set(words)
    # features['F9'] = len(V)/N
    # #N/ کلمات طولانی (بزرگ تر از 3 نویسه)
    # long_words = [w for w in words if len(w)>=3]
    # features['F10'] = len(long_words)/N
    # #N/کلمات کوچک تر از 2 نویسه
    # short_words = [w for w in words if len(w)<=2]
    # features['F11'] = len(short_words)/N
    # #N/کلمات 1 تکراره
    # counts = Counter(words)
    # unique_words = [w for w in words if counts[w]==1]
    # features['F12'] = len(unique_words)
    # #N/کلمات 2 تکراره
    # double_words = [w for w in words if counts[w]==2]
    # features['F13'] = len(double_words)
    # # معیار k یول
    # yules_k = 10000*(-1*(1.0/N) + sum(list([(len(list(w for w in V if counts[w]==i)))*((i/N)**2) for i in range(1,len(V)+1)])))
    # features['F14'] =yules_k
    # # معیار D سیمپسون
    # simpsons_d = sum((len(list(w for w in V if counts[w]==i)))*(i/N)*((i-1)/(N-1)) for i in range(1,len(V)))
    # features['F15'] = simpsons_d
    # # معیار S سیشل 
    # sichels_s = len(double_words)/len(V)
    # features['F16'] = sichels_s
    # # معیار R هونور
    # delimiter = 1 - len(unique_words)/len(V)
    # if delimiter == 0:
    #     delimiter = 0.0001
    # honores_R = (100 * math.log(N))/(delimiter)
    # features['F17'] = honores_R
    # #معیار انتروپی
    # entorpy = sum((len(list(w for w in V if counts[w]==i)))*(i/N)*(-1*math.log(i/N)) for i in range(1,len(V)))
    # features['F50'] = entorpy

    # # ویژگی های نحوی
    # # C/تعداد کاما
    # features['F18'] = text.count('،')/C
    # # C/تعداد نقطه
    # features['F19'] = text.count('.')/C
    # # C/تعداد دو نقطه
    # features['F20'] = text.count(':')/C
    # # C/تعداد سمیکلون
    # features['F21'] = text.count(';')/C
    # # C/تعداد علامت سوال
    # features['F22'] = text.count('؟')/C
    # # C/تعداد علامت تعجب
    # features['F23'] = text.count('!')/C
    # # C/تعداد علامت تعجب سه تایی
    # features['F24'] = text.count('!!!')/C

    # # ویژگی های ساختاری
    # # تعداد کل خط
    # features['F25'] = text.count('\n')
    # # تعداد کل جملات = S
    # S = len(sentences)
    # features['F26'] = S
    # # تعداد کلمه در هر جمله (میانگین)
    # features['F27'] = N/S


    # empty_lines = text.replace(" ", "").count('\n\n')
    # # print('empty lines: {}'.format(empty_lines))
    # # کل خطوط / تعداد خطوط خالی
    # total_lines = text.replace(" " , "").count('\n') + 1
    # features['F28'] = empty_lines/total_lines
    # # میانگین خطوط غیر خالی
    # # features['F29'] = sum([len(sentence) for sentence in sentences_list if len(sentence) > 1]) / (S - empty_lines)
    # features['F29'] = len(normalized_text)/(total_lines-empty_lines)

    # # ویژگی های کلمات دستوری
    # #N/تعداد ضمیر فاعلی
    # features['F30'] = utils.count_chars(text, config.subjective_pronounce)
    # for i in range(len(config.subjective_pronounce)):
    #     id = 'F30-' + str(i+1)
    #     features[id] = text.count(config.subjective_pronounce[i])
    # #N/تعداد ضمایر پرسشی
    # features['F31'] = utils.count_chars(text, config.question)
    # for i in range(len(config.question)):
    #     id = 'F31-' + str(i+1)
    #     features[id] = text.count(config.question[i])
    # #N/تعداد حرف ربط
    # features['F32'] = utils.count_chars(text, config.conjunctions)
    # for i in range(len(config.conjunctions)):
    #     id = 'F32-' + str(i+1)
    #     features[id] = text.count(config.conjunctions[i])
    # #N/حرف ربط گروهی
    # features['F33'] = utils.count_chars(text, config.group_pro)
    # for i in range(len(config.group_pro)):
    #     id = 'F33-' + str(i+1)
    #     features[id] = text.count(config.group_pro[i])
    # #N/صوت
    # features['F34'] = utils.count_chars(text, config.sounds)
    # for i in range(len(config.sounds)):
    #     id = 'F34-' + str(i+1)
    #     features[id] = text.count(config.sounds[i])
    # #N/حرف اضافه
    # features['F35'] = tags_list.count('P') + tags_list.count('POSTP')
        
    #  # نشانه های زبانی - روانی
    # pos_words, neg_words = data_loader.load_positive_negative_words(
    #     positive_words_path = data_path.POSITIVE_WORDS_PATH,
    #     negative_words_path = data_path.NEGATIVE_WORDS_PATH
    # )
    # # صفات مثبت
    # features['F36'] = utils.count_chars(text, pos_words)
    # # صفات منفی 
    # features['F37'] = utils.count_chars(text, neg_words)
    # # رنگ‌ها
    # features['F38'] = utils.count_chars(text, config.colors)
    # # کلمات رکیک
    # features['F39'] = utils.count_chars(text, config.rakik)
    # # تعداد صفات
    # features['F40'] = tags_list.count('AJ')
    # # تعداد قیود
    # features['F41'] = tags_list.count('ADV')
    # # تعداد ضمایر
    # features['F42'] = tags_list.count('PRO')
    # # نسبت ضمایر
    
    # # شک و تردید
    # features['F47'] = utils.count_chars(text, config.doubt_phrase)
    # # قطعیت
    # features['F48'] = utils.count_chars(text, config.certain_phrase)

    return features



def generate_features(dataset, label, add_text_dependent_features=True, add_word_dependent_features=True, add_syntactically_features=True,
                        add_structural_features=True, add_psycho_features=True, add_gramatical_features=True, use_gensim=True):
    features = []
    for key, data in dataset.items():
        data_features = extract_feature(data)
        data_features['Label'] = label
        data_features['number'] = key
        print('Data {} features imported'.format(key))
        features.append(data_features)
    return pd.DataFrame(features)