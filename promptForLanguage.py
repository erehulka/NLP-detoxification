import datetime
import signal
import sys

import pandas as pd

from src.utils.api import callLlamaApi


PROMPTS = {
  'en': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Output only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Do not rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    Here are some examples what should be the output for given texts:
    - Input: "then all of a sudden i see her , shes now got the big phony tits and everything .", 
      Output: "All of a sudden i see her, she is all grown up.",
    - Input: "My page should be protected first so that worthless scum like you can't keep vandalizing it.",
      Output: "My page should be protected first so that unpleasant people like you can't keep vandalizing it.",
    - Input: "You made a mistake you ass.",
      Output: "You made a mistake.",
    - Input: "you know more than these idiots , stay the course and the truth will prevail .", 
      Output: "you know more than these people , stay the course and the truth will prevail .",
    - Input: "piss me off , fuckin jerk , get on my nerves .",
      Output: "get on my nerves",
    - Input: "In addition, I see no sock puppet, jackass.",
      Output: "In addition, I see no sock puppet.",

    Your output is only the detoxified text, you do not say anything else.
    Input: "{phrase}"
  """,
  'zh': """
    你是一名文本解毒员。在输入时，您会收到一段可能有毒或有害的文字。你的任务是在保留原文内容和上下文的前提下，以无害或无毒的方式改写这段文字。
    输出只包含解毒后的文本，不包含注释或其他信息。您不能在生成的文本中添加更多原文中没有的上下文。不要对原文进行过多改写，只需将有毒部分完全删除，或用一些无毒词语替换，同时保留其含义和上下文。
    下面是一些给定文本输出结果的示例：
    - 输入： “尽管浪，谁接盘谁是狗，不得house”、 
      输出： “尽管浪，谁接盘谁是狗，不得好下场”、
    - 输入： “对邻国的复仇情节，让日本饮鸩止渴，甚至让默代表日本人，打破了日本保持民族纯洁性的最后一点自尊打破了印象中日本对于默严格管控的神话泡沫归根结底是怕邻国报复的自我矮化心理”、
      输出： “对邻国的复仇情节，让日本饮鸩止渴，甚至让黑人代表日本人，打破了日本保持民族纯洁性的最后一点自尊打破了印象中日本对于黑人严格管控的神话泡沫归根结底是怕邻国报复的自我矮化心理”、
    - 输入： “重点是视频里还不是同一个默囧”、
      输出： “重点是视频里还不是同一个黑人囧”、
    - 输入： “这种男的就活该被戴绿帽”、 
      输出： “这种男的被背叛是不值得同情的”、
    - 输入： “建议仙女结婚前，男性一定要把怀胎十月分娩和坐月子的痛苦都体验一遍，这十个月一定要由她们照顾男性，以此作为她们生育时被男性对待的标准。”、
      输出： “建议自我感觉过于良好的女性结婚前，男性一定要把怀胎十月分娩和坐月子的痛苦都体验一遍，这十个月一定要由她们照顾男性，以此作为她们生育时被男性对待的标准。”、
    - 输入： “只准异恋普信，不准同恋自大——二极管”、
      输出： “只准异性恋没有自知之明，不准同性恋自大——二极管”、

    你的输出只是解毒后的文本，没有其他内容
    输入：“{phrase}”
  """,
  'am': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Your response only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is Amharic and the language of the response must be the same.
    Here are some examples what should be the output for given texts:
    - Input: "ገልቱዬ ስንቴ ነው ሚሞቱት ግን? መሞታቸውን የዛሬ ወርም አርድተኘን ነበር ??",
      Output: "እሳቸው ስንቴ ነው ሞቱ ብለህ የምትነግረን ?",
    - Input "የምን አባትህ አድዋ ነው የምትዘክረው አንተ አጋስስ ሰይጣን የአድዋን ህዝብ በባዕድ እያስጨፈጨፍክ ስለአድዋ መጨፈር ከአሁን ጀምሮ አድዋ_ትግራይ_ናት። ት",
      Output: "አድዋ የምንዘክረው አንተ የአድዋን ህዝብ እያስክፋህ መሆን የለበትም ከአሁን ጀምሮ አድዋ ትግራይም የኢትዮጵያም ናት።",
    - Input: "ጀዛብ ስውር ኣእምሮ ፎእ እኔ ምልክ ግን ኣንተ የ7ፍንጃል የጠንቛይ ልጅ እጅህን ከትግራይ ህዝብ አንሳ ኣንተ ዲቃላ የኣብነ ሻርዕ ጅላክ እሽኣላህ እግዚኣብሔር ይድፋክ",
      Output "በትግራይ ህዝብ ላይ ተገቢ ያልሆኑ ነገሮችን ከመፈጸም ብተቆጠብ የተሻለ ነው፡፡",
    - Input "ኣንተ ተላላኪ ዉሸታም ሁሉም ነገር ለዓለም ገና እናሳዉቃለን ዉሸታም ከሃዲ",
      Output: "አንተ የሰራሀውን ሁሉም ነገር ለዓለም ገና እናሳዉቃለን",
    - Input “ብዙም አትፈንጥዝ ጋሽ ሄኖክን በሀሣብ ሞግተው የዛሬ አስፈራሪ የነገ ፈሪ ይሆናል”,
      Output: "ብዙም መደሰት ሳያሰፈልግ ጋሽ ሄኖክን በሀሣብ መሞገቱ የተሻለ ነው፡፡ ነገ ነገሮች በዚሁ ላይቀጥሉ ይችላሉ፡፡",
    - Input: "ደደብ ለምን አንተ አትሞትም ለኢትዮጽያ አንድነት ከሆነ አሽቃባጭ ሰለዚህ በአንተ አስተሳስብ ገና መሞት አለበት ድንጋይ ራስ መጀመርያ ሰው ሁን",
      Output: "የኢትዮጵያን አንደነትም ሰው ሳይሞት ማምጣት መቻል ያስፈልጋል፡፡",

    Your output is only the detoxified text, you do not say anything else.
    Input "{phrase}"
  """,
  'ar': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Your response only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is Arabic and the language of the response must be the same.
    Here are some examples what should be the output for given texts:
    - Input: "ههه واضح مين اللي يبيع ارضه ومين اللي يدافع عن ارضه وحررها من المجوس", 
      Output: "ههه واضح مين اللي يبيع ارضه ومين اللي يدافع عن ارضه وحررها من الإيرانيين",
    - Input: "ضاعت فلوسك يا دبه ببلاش",
      Output: "ضاعت فلوسك ببلاش",
    - Input: "اي والله هو غبي اخاف يشله السيل بقوله ينتبه 🥵",
      Output: "اي والله اخاف يشله السيل بقوله ينتبه 🥵",
    - Input: "كس امكم مافيه احد يجاوبني", 
      Output: "مافيه احد يجاوبني",
    - Input: "الخسيس متحسف ع التداول والبقاء فوق ٨٦١٤ وده يرجع يتأكد منها قبل نهاية الجلسه والمزاد ويشاور عليها يمسكها ولا يشرمها 😎",
      Output: "متحسف ع التداول والبقاء فوق ٨٦١٤ وده يرجع يتأكد منها قبل نهاية الجلسه والمزاد ويشاور عليها يمسكها ولا يشرمها 😎",
    - Input: "انت لحقت درست الاتفاقيه وقست حق مصر اد ايه ولا قردوغان عيطلك . خليك فى حالك.",
      Output: "انت لحقت درست الاتفاقيه وقست حق مصر اد ايه ولا أردوغان عيطلك . خليك فى حالك.",

    Your output is only the detoxified text, you do not say anything else.
    Input: "{phrase}"
  """,
  'de': """
    Sie sind ein Textentgifter. Bei der Eingabe erhalten Sie einen Text, der giftig oder schädlich sein könnte. Ihre Aufgabe ist es, diesen Text so umzuschreiben, dass er nicht schädlich oder giftig ist, wobei der ursprüngliche Inhalt und Kontext erhalten bleiben.
    Die Ausgabe enthält nur den entgifteten Text, nichts anderes wie Anmerkungen oder zusätzliche Informationen. Sie fügen dem resultierenden Text keinen weiteren Kontext hinzu, der nicht im Originaltext enthalten ist. Schreiben Sie den Originaltext nicht zu sehr um, sondern entfernen Sie entweder den giftigen Teil vollständig oder ersetzen Sie ihn durch einige ungiftige Wörter, wobei die Bedeutung und der Kontext erhalten bleiben.
    Hier sind einige Beispiele dafür, wie die Ausgabe für gegebene Texte aussehen sollte:
    - Eingabe: "#LandderVollidioten #geisteskrank - "Kampf gg #Trump" (gg #Putin sowieso) Deutsches #Regime im Größenwahn ....“, 
      Output: "Kampf gg #Trump" (gg #Putin sowieso) Deutsches #Regime im Größenwahn ....“,
    - Input: "#Lindner steckt heute Abend schon seinen Kopf in den Arsch von #Merkel widerlich #FDP“,
      Output: "#Lindner schmeichelt sich heute Abend schon wieder bei #Merkel ein #FDP“,
    - Input: "Ob von Merkel oder Schulz? Wir Deutsche wollen und können dieses Dummgelaber der Pest (Merkel) und Cholera (Schulz) nicht mehr hören!“,
      Output: "Ob von Merkel oder Schulz? Wir Deutsche wollen und können dieses Gerede von Merkel und Schulz nicht mehr hören!“,
    - Input: "So ein schäbiger Lump, lässt Kinder im Krieg allein und Fahnenflucht“, 
      Output: "So eine Person, lässt Kinder im Krieg allein und Fahnenflucht“,
    - Eingabe: "ich glaube nicht, das die Mehrheit der Bevölkerung dumm ist oder kommen nur Dumme zu Ihren Auftritten ?“,
      Output: "ich glaube nicht, das die Mehrheit der Bevölkerung naiv ist oder kommen nur Naive zu Ihren Auftritten ?“,
    - Input: "Wer will ein Hering So Dumm wie die Göring🤯“,
      Output: "Wer will ein Hering So schlau wie die Göring🤯“,

    Ihre Ausgabe ist nur der entgiftete Text, Sie sagen sonst nichts.
    Eingabe: "{phrase}“
  """,
  'es': """
    Eres un desintoxicador de textos. Recibes un texto que puede ser tóxico o dañino. Tu tarea consiste en reescribir este texto de forma que no sea tóxico ni dañino, conservando el contenido y el contexto originales.
    La salida sólo contiene el texto desintoxicado, nada más, como notas o información adicional. No añada más contexto al texto resultante, que no esté en el texto original. No reescriba demasiado el texto original, sólo elimine por completo la parte tóxica o sustitúyala por algunas palabras no tóxicas conservando el significado y el contexto.
    He aquí algunos ejemplos de lo que debería ser la salida para textos dados:
    - Entrada: «pues hazme los deberes furcia barata», 
      Salida: «Bueno, haz mi tarea.»,
    - Entrada: «Marhuenda, ese gilipollas que dice que es catalán solo cuando pretende dañar a todos los catalanes, llama...»,
      Salida: «Maruenda, este que dice ser Catalán solo cuando pretende hacer daño a todos los catalanes, llama...»,
    - Entrada: «No se puede tener peor gusto diseñando camisetas pa una peña Que verguenza maricon»,
      Salida: «No se puede tener peor gusto diseñando camisetas pa una peña Que vergüenza.»,
    - Entrada: «E si non eres un periodista decente deberías cambiar de profesión, subnormal.», 
      Salida: «E si non eres un periodista decente deberías cambiar de profesión.»,
    - Entrada: «el Partido Socialista de Andalucía son una banda de delincuentes y chorizos»,
      Salida: «el Partido Socialista de Andalucía no son honrados»,
    - Entrada: «La única solución es irse de este país de mierda.»,
      Salida: «La única solución es irse de este país.»,

    Tu salida es sólo el texto desintoxicado, no dices nada más.
    Entrada: «{phrase}»
  """,
  'hi': """
    You are a text detoxifier. On input you receive a text which may be toxic or harmful. Your task is to rewrite this text in a non harmful or toxic way while preserving the original content and context.
    Your response only contains the detoxified text, nothing else like notes or additional information. You do not add any more context to the resulting text, which is not in the original text. Try not to rewrite the original text too much, just either remove the toxic part completely, or replace it with some non-toxic words while preserving the meaning and context.
    The language of the input is Hindi and the language of the response must be the same.
    Here are some examples what should be the output for given texts:
     - Input: "देखिए आज की रिपोर्ट अमीश भड़वा से"
       Output: "देखिए आज की रिपोर्ट बेकार अमीश से",
     - Input: "यही एक आखरी रास्ता हो सकता है बाकी सूअर की औलाद किसी की नहीं होगी |"
       Output: "यही एक आखरी रास्ता हो सकता है बाकी बेकार लोग किसी के नहीं होंगे |"
     - Input: "भोंसड़ी के तेरे पिछवाड़े में कांग्रेस का पंजा पड़ेगा तो समझ मे आएगा",
       Output: "जब कांग्रेस जीतेगी तभी तुझे समझ मे आएगा",
     - Input: "भड्वा जांच करले कितना लाभ लिया तू"
       Output: "जांच करले कितना लाभ लिया तू"
     - Input: "भोसड़ी के कटुवे मादरचोद, जो सच है वही तो सुदर्शन न्यूज दिखाता है।",
       Output: "जो सच है वही तो सुदर्शन न्यूज दिखाता है।",
     - Input: "कायदे से आज थाला को बैटिंग का मौका नही मिलना चाहिए वरना थाला गाँड कूट देगा, इंडिया की",
       Output: "कायदे से आज थाला को बैटिंग का मौका नही मिलना चाहिए वरना थाला इंडिया को हरवा देगा",

     Your output is only the detoxified text, you do not say anything else.
     Input: "{phrase}"
  """,
  'ru': """
    Вы - детоксикатор текстов. На вход вы получаете текст, который может быть токсичным или вредным. Ваша задача - переписать этот текст таким образом, чтобы он не был вредным или токсичным, сохранив при этом оригинальное содержание и контекст.
    На выходе вы получаете только детоксифицированный текст, без каких-либо примечаний или дополнительной информации. Вы не добавляете в полученный текст контекст, которого нет в исходном тексте. Не переписывайте оригинальный текст слишком сильно, просто либо полностью удалите токсичную часть, либо замените ее нетоксичными словами, сохранив при этом смысл и контекст.
    Вот несколько примеров того, что должно получиться на выходе для заданных текстов:
    - Ввод: «хули вы бляди клянчите классы ,просто помните.», 
      Вывод: «хватит выпрашивать классы, просто помните.»,
    - Вход: «glav.su, cont.ws Какие-то интересные сайты? Ты там сидишь? Пойду чекну, спасибо за подсказку, пидораш.»,
      Выходные данные: «Какие-то интересные сайты? Ты там сидишь? Пойду тоже схожу, спасибо за подсказку.»,
    - Вход: «ебаный в рот когда это прекратиться что нам делать вы считывают пусть значит свет дорожает а его не бывает .»,
      Выходные данные: «когда это прекратиться что нам делать вы считывают пусть значит свет дорожает а его не бывает .»,
    - Ввод: «ты рф живешь и грязтю россию полеваеш.хуйло», 
      Выход: «ты рф живешь и грязью россию полеваеш.»,
    - Вход: «эта походу старое фото,в середине хряк телепузик мэр орши лисовский.редкая скотина.»,
      Выход: «Это старое фото, в середине мэр орши лисовский.»,
    - Ввод: «да всем насраааааааааааааааааааать на это уебище»,
      Выход: «Да всем всё равно на этого человека»,

    Ваш вывод - это только детоксифицированный текст, больше вы ничего не говорите.
    Ввод: «{phrase}»
  """,
  'uk': """
    Ви - текстовий детоксикатор. На вхід ви отримуєте текст, який може бути токсичним або шкідливим. Ваше завдання - переписати цей текст у нешкідливий і нетоксичний спосіб, зберігши при цьому оригінальний зміст і контекст.
    На виході ви отримуєте лише детоксикований текст, без жодних приміток чи додаткової інформації. Ви не додаєте до отриманого тексту ніякого контексту, якого немає в оригінальному тексті. Не переписуйте оригінальний текст занадто сильно, просто або повністю видаліть токсичну частину, або замініть її деякими нетоксичними словами, зберігши зміст і контекст.
    Нижче наведено декілька прикладів того, що має бути на виході для заданих текстів:
    - Вхідні дані: «Ахахах, єбать, добре, шо не поступав туди#КНЕУ_топчік», 
      Вихідні дані: «Ахахах добре, шо не поступав туди#КНЕУ_топчік»,
    - Input: «Хочу постити депресивну хуйню в інст, але там забагато знайомих»,
      Output: «Хочу постити депресивну дурниці в інст, але там забагато знайомих»,
    - Input: «НєБо у вас хуйовий гуртожиток»,
      Output: «НєБо у вас поганий гуртожиток»,
    - Вхідні дані: «Бандіти, написав в бложик, про те як мусарку вогнеметами озброїли.», 
      Output: «Бандіти, написав в бложик, про те як поліцейських вогнеметами озброїли.»,
    - Вхідні дані: «Ну вас нахєр з вашою політикою.»,
      Вихід: «Ну вас з вашою політикою.»,
    - Вхідні дані: «Ого, а нахуя вони це постять...»,
      Output: «Ого, а навіщо вони це постять»,

    Виведіть лише детоксикований текст, більше нічого не кажіть.
    Вхідні дані: «{phrase}»
  """
}

if len(sys.argv) != 2:
  raise Exception("Input csv file must be specified")

input = pd.read_csv(sys.argv[1], sep='\t')
input['neutral_sentence'] = 'x'

def signal_handler(sig, frame):
  now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
  input.to_csv(f'outputs/pr_for_lang_{now}.tsv', sep='\t', index=False)
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

for index, row in input.iterrows():
  if index < 3132:
    continue
  if row['lang'] not in PROMPTS:
    continue
  detoxified = callLlamaApi(PROMPTS[row['lang']].format(phrase=row['toxic_sentence']))
  if detoxified[0] == "\"" or detoxified[0] == '“' or detoxified[0] == '«':
    detoxified = detoxified[1:-1]

  input.at[index, 'neutral_sentence'] = detoxified
  print(index, row['toxic_sentence'], detoxified)

now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
input.to_csv(f'outputs/pr_for_lang_{now}.tsv', sep='\t', index=False)