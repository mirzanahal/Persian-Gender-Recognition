
FEMALE_LABEL = 0
MALE_LABEL = 1

MODEL_SCORE_METRICS = ['precision', 'recall', 'f1', 'accuracy']


colors = ['خاکستری' , 'بنفش' , 'برگ سنجدی' , 'ارغوانی', 'آلبالویی' , 'آبی' , 'پوست پیازی' , 'پسته‌ای', 'پرکلاغی' ,
          'سرخابی' , 'سبز' , 'زیتونی' , 'زرد' , 'دوغی' , 'دودی' , 'دریایی', 'خرمایی' , 'خاکی' , 
          'فیروزه‌ای' , 'عدسی', 'طلایی' , 'صورتی' , 'شبدیز' , 'سیاه', 'سفید' , 'سرمه‌ای' , 'سرخ' ,
          'قرمز', 'یاسی' , 'نقره‌ای' , 'نخودی' , 'نارنجی' , 'موشی' , 'مغز' , 'ماشی' , 'فیلی',
          'فیروزه ای' , 'سرمه ای' , 'زرشکی' , 'بژ' , 'طلائی' , 'ماشی' , 'کرم' , 'نقره ای' , 'شرابی',
          'عنابی' , 'ارغوانی' , 'شیری' , 'لیمویی' , 'هلویی' , 'نخودی' , 'خاکی' , 'مغز پسته ای' , 'مغزپسته‌ای',
          'یشمی' , 'دودی' , 'نیلی' , 'لاجوردی' , 'آبی نفتی' , 'بادمجانی' , 'گندمی' , 'خردلی' , 'عسلی',
          'استخوانی' , 'خاکستری' , '' , '' , '' , '' , '' , '' , '',]

rakik = [ "آب کیر", "آشغال", "آلت تناسلی", "آلت",
          "ابله", "ابن یزید", "احمق", "اسب", "اسبی", "اسکل", "اسکل", "اسگل", "اسگول", "الاغ", "الاق", "انگل", "انی", "انی", "اوسکل", "اوسکل", "اوسگل", "اوصکل",
          "اوصگل", "ب ک", "باسن", "بخورش", "بدبخت", "بمال", "به تخمم", "به چپم", "به کیرم", "بپرروش", "بپرسرش", "بچه کونی", "بکارت", "بکن بکن", "بکن توش",
          "بکن", "بکنش", "بکنمت", "بی آبرو", "بی خایه", "بی شرف", "بی شعور", "بی عفت", "بی غیرت", "بی ناموس", "بی ناموس", "بی پدر", "بیابخورش", "بیشعور",
          "بیناموس", "پفیوز", "تخم سگ", "تخمی", "ترک", "توله سگ", "جاکش", "جاکش", "جاکش", "جلق زدن", "جنده خانه", "جنده", "جنسی", "جوون", "جکس", "جیندا",
          "حرومزاده", "حشر", "حشری شدن", "حشری", "حیوانی", "خارکس ده", "خارکسده", "خارکسده", "خارکسّه", "خانم جنده", "خایه خور", "خایه مال", "خایه مال",
          "خایه", "خر", "خرفت", "خری", "خز", "خفه خون", "خفه شو", "خفه", "خنگ", "خواهرجنده", "خی کاس", "داف ناز", "داف", "داگ استایل", "دخترجنده",
          "دخترقرتی", "درازگوش", "دله", "دهن سرویس", "دهن گاییده", "دهنت سرویس", "دهنتوببند", "دوجنسه", "دوست دختر", "دوست پسر", "دول", "دکل", "دیوث",
          "دیوس خان", "دیوس", "دیوص", "رشتی", "ریدن", "ریدی", "زارت", "زباله", "زرنزن", "زن جنده", "زن جنده", "زن کاسده", "زنا زاده", "زنا", "زنازاده",
          "زنتو", "زنشو", "زنیکه", "سادیسمی", "ساک بزن", "ساک", "ساکونی", "سرخور", "سرکیر", "سسکی", "سوراخ کون", "سوراخ کون", "سولاخ", "سکس چت", "سکس", 
          "سکسی باش", "سکسی", "سکسیم", "سکسیی", "سگ تو روحت", "سگ دهن", "سگ صفت", "سگ پدر", "سگی", "سیکتیر", "شاسگول", "شاش", "شق کردن", "شل مغز",
          "شنگول", "شهوتی", "شورتم ماسکت", "صیغه ای", "صیک", "عرب", "عرق خور", "عمتو", "عمه ننه", "عن تر", "عن", "عن", "عنتر", "عوضی", "غرمساق",
          "غرمصاق", "فاحشه خانم", "فاحشه", "فارس", "فاک فیس", "فیلم سوپر", "قرتی", "قرمساق", "قرمصاق", "قس", "لا پا", "لاس", "لاش گوشت", "لاشی", "لاشی",
          "لامصب", "لاکونی", "لجن", "لخت", "لختی", "لر", "لز", "مادر جنده", "مادرجنده", "مادرسگ", "مادرقهوه", "مادرکونی", "مالوندن", "ماچ کردنی", "ماچ",
          "مرتیکه", "مردیکه", "مرض داری", "مرضداری", "مشروب", "ملنگ", "ممه خور", "ممه", "منگل", "میخوریش", "نرکده", "نعشه", "نکبت", "نگاییدم", "هیز",
          "ولدزنا", "پدر سوخته", "پدر سگ", "پدر صلواتی", "پدرسگ", "پریود", "پستان", "پسون", "پشمام", "پفیوز", "پلشت", "پورن", "پپه", "چاغال", "چاقال",
          "چس خور", "چس", "کاسکش", "کث لیس", "کث", "کثافت", "کثافط", "کردن", "کردنی", "کرم", "کس خل", "کس خور", "کس خیس", "کس دادن", "کس لیس", "کس لیس",
          "کس لیسیدن", "کس ننت", "کس و کیر", "کس کردن", "کس کش", "کس", "کسخل", "کسشعر", "کسکش", "کسکیر", "کص خل", "کص لیس", "کص", "کصافت", "کصافط",
          "کصخل", "کصکش", "کلفت", "کله کیری", "کله کیری", "کوث لیس", "کوس خل", "کوس خور", "کوس لیس", "کوس", "کوص خل", "کوص لیس", "کوص", "کون تپل",
          "کون ده", "کون سوراخ", "کون پنیر", "کون گنده", "کون", "کونده خار", "کونده خوار", "کونده", "کونشو", "کونن", "کونی", "کونی", "کیر", "کیردراز",
          "کیردوس", "کیرر", "کیرمکیدن", "کیرناز", "کیروکس", "کیروکس", "کیری", "گاو", "گاوی", "گاگول", "گایدن", "گایدی", "گاییدن", "گردن دراز",
          "گشاد", "گنده گوز", "گه", "گهی", "گوز باقالي", "گوز", "گوزو", "گوزو", "گوسفند", "گوش دراز", "گوه", "گوه", "گی زن", "گیخوار", "یبن زنا"
    ]

question = ['چی' , 'چرا' , 'چه' , 'چندمین', 'چند' , 'ایا' , 'چطور' , 'چگونه','آیا' , 
            'کجایی' , 'کجائی' , 'کجا','کدامیک' , 'کدامین' , 'کدام' , 'کی', 'کي' , 'که']

sounds = ['براوو' , 'باللّه' , 'به‌به' , 'بادا', 'بادا' , 'آخه' , 'آمین' , 'عجبا', 'احسنت' ,
          'زکی' , 'زهازه' , 'یا', 'خوشا' , 'وای' , 'وای' , 'وااسفا', 'وا' , 'اوخ' ,
          'جانمی' , 'هورا', 'هیهات' , 'حیف' , 'هان' , 'ای', 'اوا' , 'ان‌شاالله' ,
          'زرت' , 'دالی', 'کریما' , 'پیشت' , 'مرده‌باد' , 'مرسی', 'مرحبا' , 'لبیک' ,
          'کریما' , 'دردا','سک سک' , 'آفرین' , 'کاش' , 'دریغا', 'تبارک الله' , 'آهای']

numbers = ['0' , '1' , '2', '3' , '4' , '5', '6' , '7' , '8', '9' ,
           '۰' , '۱', '۲' , '۳' , '۴', '۵' , '۶' , '۷', '۸' , '۹',
           'یک' , 'دو', 'سه' , 'چهار' , 'پنج', 'شش' , 'هفت' , 'هشت', 'نه' , 'صفر',
           'یه' ]

special_chars = ['!' , '#' , '?' , '%' , '*' , '(' , ')' , ':' , '-' , '+' , '=' , '/' , '.', ',']

finished_chars = ['!', '?', '.']

conjunctions = ['چون' , 'گرچه' , 'هکذا' , 'هم' , 'هرچند' , 'اگر' , 'ولیکن' , 'ولو' , 'یا' , 'یعنی' , 'زیرا' ,
                 'اما' , 'آن‌چنان‌که' , 'آنچنانکه' , 'چه' , 'که' , 'و' , 'تا' , 'لیکن' , 'لکن' ]

doubt_phrase = ['ظن' , 'فرضاً' , 'فرض' , 'احتمالاً' , 'احتمالا' , 'محتمله' , 'محتمل' , 'به نظرم' ,
                'به نظر من' , 'شاید' , 'احتمالا' , 'گویا' , 'ممکنه' , 'احتمالا' , 'ممکن' , 'گویا' , 'ممکن' , 'یحتمل' , 'حدس' , 'گمان' ]

certain_phrase = ['هیچگاه' , 'هیچ موقع' , 'هیچ وقت' , 'هیچوقت' , 'حاشا' , 'هیچ' , 'هیچی' , 'بی شبهه' , 'بدون شبهه' ,
                  'بدون شک' , 'بی شک' , 'مسلماً' ,      'یقیناً' , 'بیقین' , 'به یقین' , 'مطمئناً' , 'مطمئن' , 'مطمئنم' , 'همواره' , 'پیوسته' ,
                  'همیشه' , 'واقعاً' , 'ابداً' , 'هرگز' , 'ابدا' , 'یقینا' , 'مسلما' , 'مطمئنا' , 'همه وقت' , 'همه کس' , 'همه' , 'تمام' ]

subjective_pronounce = ['آن'  , 'اونا' , 'اون' , 'آن‌ها' , 'شما' , 'تو' , 'او' , 'ما' , 'من' , 'اون‌ها']

alphabet = ["آ", "ا" , "ب" ,"پ" , "ت" , "ث" , "ج" , "چ" , "ح",
            "خ", "د" , "ذ" , "ر" , "ز" , "ژ" , "س" , "ش",
            "ص", "ض" , "ط", "ظ", "ع" , "غ" , "ف" , "ق",
            "ک" , "گ" , "ل", "م" , "ن" , "و" , "ه" , "ی" ]

group_pro = ["وندر", "بدون‌آن‌که" , "باوجوداین‌که" , "برای‌آن‌که" , "بااین‌وصف" , "بعدازآن‌که" , "باآن‌که" , "ازاین‌رهگذر",
             "ازاین‌جهت", "به‌همین‌جهت" , "به‌علت‌آن‌که" , "به‌عبارتی‌دیگر" , "بعبارت‌دیگر" , "بدین‌جهت" , "ازاین‌رو" , "ازاین‌گذشته",
             "به‌این‌علت‌که", "به‌این‌جهت‌که" , "به‌همین‌ترتیب" , "به‌همین‌منظور" , "به‌همین‌علت" , "به‌بیان‌دیگر" , "باوجوداین‌" , "ازهمین‌رو",
             "به‌این‌خاطرکه", "به‌سبب‌آن‌که" , "به‌خاطرآن‌که" , "بیشترازآن‌که" , "به‌قدری‌که" , "به‌محض‌این‌که" , "بهترازآن‌که" , "به‌همین‌نحو",
             "چون‌که", "زانکه" , "تاحدی‌که" , "روزی‌که" , "پیش‌ازآن‌که" , "نه‌چندان" , "هنگامی‌که" , "هرچندکه",
             "همین‌که", "قبل‌ازآن‌که" , "پس‌ازآن‌که" , "مگراین‌که" , "هرجاکه" , "حالاکه" , "درغیراین‌صورت" , "درهمین‌حال",
             "درهمان‌حال", "وقتی‌که" , "کمترازآن‌که" , "کمترازآن‌که" , "همان‌اندازه‌که" , "درهمین‌رابطه" , "درعین‌حال" , "بی‌آن‌که",
             "زمانی‌که", "جایی‌که" , "درصورتی‌که" , "درحالی‌که" , "چه‌این‌که" , "به‌مجرداین‌که" , "به‌این‌سبب‌که" , "به‌جهت‌آن‌که",
             "وز" , "نه‌تنها" , "کاندر", "کز" , "چندان‌که" , "چنان‌که" , "چراکه" , "افزون‌براین" , "ازآن‌رو" , "آنجاکه" , "اگرچه" , "اگرنه" ]