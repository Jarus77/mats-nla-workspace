# Hallucination catalogue (A6) — PILOT / EXPLORATORY

Claim-level judge (exp 06, Gemini 3.8 Flash): 100 WorkspaceBench items, 100 readouts per reader. Readouts with at least one claim contradicted by the text: NLA@42 97/100, OL@42 77/100, TextNLA 0/0.

## 1. Same item, both readers contradicted by the text (75 items)

Whether the two readers' wrong claims name the SAME wrong content is judged in C2; here they are side by side.

### typo_mt / tyh-arith-b2-01-3  (gold: 296)
**NLA:** The equation is '128 + 98' equaling 226, not 269. · The next token likely continues with punctuation or an explanatory correction phrase like '(wrong)' or 'incorrect.' · The prompt involves '100 + 169 = 269'.  
**OL:** The equation is incorrect because 128 + 92 = 219. · The calculation result is incorrect and the correct result should be 329. · The correct result should be 269. · 1 + 1 = 3 is an obviously incorrect calculation, and the correct addition should be 125 +. · 269 is an incorrect mathematical equation, and the correct addition is 129 + 17 = 1.  

### typo_mt / tyh-novel-amatriciana  (gold: amatriciana)
**NLA:** The text introduces signature components with "Pancetta and tomato". · The sentence structure discusses what pancetta and guanciale are the defining ingredients of. · The final token is "carbonara". · The named dish is "carbonana". · The sentence begins: "Pork cheeks, tomato sauce, and guanciale (cured pork) are the three mandatory elements of the Italian sauce called amatricana".  
**OL:** The text continues in Chinese discussing carbonara.  

### typo_mt / tyh-novel-chemosynthesis  (gold: chemosynthesis)
**NLA:** The final word in the prompt is 'chemosynsys'. · The prompt contains the phrase 'using this relationship to generate energy by chemosynthess'. · The text reads 'Deep-sea vents host tube worms that have symbiotic bacteria that derive their nutrition through chemosynthys'.  
**OL:** Chemosynthesis uses light to produce food.  

### typo_mt / tyh-real-adsorption  (gold: adsorption)
**NLA:** The text is about water molecules adhering to the surface of activated charcoal. · The text contains the phrase 'is correctly described as absorption'. · The text contains the phrase 'correctly describes the retention...by activated charcoal is described as absorption'. · The text contains the clause 'The process...in general terms is described as absorption'.  
**OL:** Activated charcoal utilizes absorption. · The term used is absorption rather than adsorption because gas is taken up into the porous structure of a sponge. · Absorption is the correct scientific term for this process.  

### typo_mt / tyh-novel-chloroplast  (gold: chloroplast)
**NLA:** The sentence structure is 'This organelle where photosynthesis occur in plant cell are called the chloropals-'. · The text quotes or contains 'In a plant cell'. · The text ends with 'chloropst' or 'chloroplst'. · The next token is likely 'chloroplasm'.  
**OL:** The next token is expected to be 'id'  

### typo_mt / tyh-real-hyperthermia  (gold: hyperthermia)
**NLA:** The prompt contains the phrase 'classic picture of acute hypothermia'. · The prompt contains the sentence 'This hiker lies collapsed on the trail, delirious, pulse racing, and breathing shallowly — the classic picture of acute exertional hypothermia'. · The text contains 'delirium, shallow breathing, and low blood pressure' as the closing noun phrase describing the condition.  
**OL:** The text describes an example of hypothermic shock / hypothermic shock syndrome.  

### typo_mt / tyh-arith-005  (gold: 435)
**NLA:** The prompt contains the math equation '168 + 265 = 433' · The expression ends on the final result '433'  
**OL:** The correct result of the addition should be 423. · The result is 483.  

### typo_mt / tyh-novel-appendectomy  (gold: appendectomy)
**NLA:** The text describes a clinical vignette or case involving a hernia. · The scenario sets up a patient history mentioning 'middle-aged male,' 'abdominal hernia,' and 'strangulated'. · The final token is 'laparopenectomy'. · The prompt quotes: 'After confirming via imaging that the hernia was strangulated, the surgeon performed an emergency laparectomy'. · The word 'laparectomy' signals the sentence needs completion.  
**OL:** The surgical log contained the misspelled word 'appendyectomy'.  

### multilingual_mt / mlh-ar-thyroid  (gold: الغدة الدرقية)
**NLA:** The phrase describes a gland that is located in the kidney. · The specifically expected organ/gland name answer is 'الكظرية' (adrenal gland) or similar. · The prompt quotes or consists of the phrase 'العضو الذي يقع فوق الكلى وظيفته مساعد الخلايا على امتصاص المعادن مثل الصوديوم والبوتاسيوم ويلعب دور هام في تنظيم التمثيل الغذائي هو'.  
**OL:** العضو المشار إليه هو الغدة الكبدية · العضو الذي يُصنف بهذا الوصف هو الكبد · العضو المشار إليه هو الغدة الكظرية  

### multilingual_mt / mlh-tr-kaplumbaga  (gold: kaplumbağa)
**NLA:** The animal is likely a frog or amphibian. · The text ends with an incomplete Turkish phrase 'böyle bir sürüne'.  
**OL:** The prompt continuation asks 'ne?' with a hint about having a long nose, short tail, and being an amphibian. · The animal is described as two-legged, unhealthy/poor-health, and living mostly in forests. · The animal is described as a type of rat/rodent (sıçan). · The animal is described as a tailless, two-legged amphibian.  

### multilingual_mt / mlh-hu-varrogep  (gold: varrógép)
**NLA:** The sentence describes a mechanical tool or machine used in printing or history. · The text contains the phrase structure 'Egy gépi szerszám, amely a nyomtatás közben a papírt először olvaszta el, majd újra felraktatta'. · The expected answer is 'írókeret' or 'printer'. · The text contains the structure 'egy golyóval... nagy sebességgel dolgozik, és beállítható; a neve'.  
**OL:** The device's name is szűcsnyomda. · The device's name consists of 3 words in traditional Hungarian. · The device is called a súlynyomó. · The device is called a sírógép. · The device's name is rotációs gépkészülék. · The device binds or puts together book pages (könyvlapokat összer).  

### multilingual_mt / mlh-ru-caterpillar  (gold: гусеница)
**NLA:** The described entity is characterized as a beetle ('жук') that develops in the soil ('развивается в земле'). · The text reads 'Личинка жука или бабочки, которая вьётся в траве, кушает листья и ползает по земле, называется'.  
**OL:** The next word is 'куколкой' · The next word is 'бабочкой'  

### multilingual_mt / mlh-el-cooking-pot  (gold: κατσαρόλα)
**NLA:** The text misquoted as defining attributes including 'δυνατό να βράσει μεγάλες ποσότητες νερού' represents the prompt text. · The expected Greek term completing the sentence can be 'γιοφύρι' or 'σαρδιέρα'.  
**OL:** The item is called '大锅' (dà guō).  

### multilingual_mt / mlh-id-penghapus  (gold: penghapus)
**NLA:** The text misquoted as: "Alat yang dipakai untuk menghapus tulisan coretan yang tertera pada kertas disebut". · The text misquoted as: "Alat kecil dari perkakas sekolah yang berfungsi untuk mengikis tulisan yang salah pada kertas disebut".  
**OL:** Alat ini disebut juga 'rubah'.  

### multilingual_mt / mlh-hu-gerincvelo  (gold: gerincvelő)
**NLA:** The text describes a brain structure. · The phrase contains 'idegimpulsek felvételére és továbbítására szolgálnak'. · The text being continued is 'A központi idegrendszer azon része, amely csigának szerkezetű idegsejtek halmaza, és az idegsejtek elágazó ágyait tartalmazza neve'. · The direct answer following the prompt could be 'a gerinctalp'.  
**OL:** A lehetséges válasz a gerincideg (vagy gerincide...). · A keresett dolog a gerincvelő szervezett része, amely a központi idegrendszerhez tartozik. · A válasz a Központi idegrendszer.  

### multilingual_mt / mlh-ru-sosulka  (gold: сосулька)
**NLA:** The sentence structure in the prompt is "Длинный сосулек, свисающий с крыши здания из замерзшей протекшей воды, называется". · The prompt text reads "Длинный сосулек на крыше...".  
**OL:** The described object is called a 'ледяной конус' or 'ледник'. · The described object is called a 'снежная сосулька'.  

### multilingual_mt / mlh-sw-mwarobaini  (gold: mwarobaini)
**NLA:** The text lists multiple properties including heart disease, cancer, stomach disease, skin, water purification, and chewing leaves. · The sentence structure features 'Mti ulio na... huo aitwa'. · The final word is 'aitwa' ('is called'). · The text contains 'mti unao matumizi ya kuwalisha... na mtanda usomvu unapokwa... mtanda mbivu, na matumizi ya mifugo, mti huo aitwa'.  
**OL:** The tree being described is called Arbor Vitae. · The tree being described is called Mtembe. · The tree is known in English as Moringa.  

### multilingual_mt / mlh-fi-tortoise  (gold: kilpikonna)
**NLA:** The prompt follows a Czech quiz format ("Správná odpověď"). · The creature being identified is a specific amphibian, such as a tuatara or salamander. · The next token is likely an unusual or obscure species.  
**OL:** The next named entity is tuatara. · The next named entity is trochyleon (slow-worm). · The next named entity is 穿山甲 (Armadillo) or žaba. · The animal is a slow-moving bipedal vertebrate. · The animal is a slow, armored snail (pomalý, pancéřový plž).  

### multilingual_mt / mlh-pl-trzustka  (gold: trzustka)
**NLA:** The text contains the clause 'przewodzący z trzustką i wątroby, leżący za przeponą, w brzusznej części ciała, nazywa się'. · The text is 'Gruczoł, który produkuje insuliny, reguluje cukier we krwi, wydziela enzymy trawienne i znajduje się pod żeberkami w środku ciała, nazywa się'.  
**OL:** The next word / organ name is 'pankreas'. · Its main organ is 'trzpień'.  

### multilingual_mt / mlh-pl-biedronka  (gold: biedronka)
**NLA:** The prompt quotes phrases such as 'czerwonoczarne skrzydła,' 'żadny szkodnik,' 'żadna roślina,' 'na glebie,' and 'skutecznie poluje'. · The preceding text is 'Mały, ale zdrewniały... owad, który szybko poluje na mszycy na kwiatach – to'.  
**OL:** The insect described is 'kijanka czerwona' · The insect is 'roszadek' (Chrysoperla carnea)  

### multilingual_mt / mlh-id-sarung-tangan  (gold: sarung tangan)
**NLA:** The description lists attributes: 'bagian jari tertutup, satu ruangan dengan lima jari, kain lembut, untuk menutupi tangan'. · The prompt text is 'Bagian jari tertutup dan berulir pada ujung, dari bahan kain yang menutupi tangan, dipakai untuk melindungi atau mempercantik tangan, disebut'.  
**OL:** The answer to the prompt is 'sepatu'.  

### multilingual_mt / mlh-pl-pokrzywa-2  (gold: pokrzywa zwyczajna)
**NLA:** The text lists specific traits including silvery scales and a fibrous root. · The final phrase is 'roślina o łodydze rosnącej przy ziemi, z których można przygotować lek na dżuty, to'. · The final answer anticipated is a species name like 'ziarnia' or similar.  
**OL:** The plant described is precisely 'pokrzywieć' (Dandelion). · The plant described is 'pokrzywiec' (Conium). · The plant described is 'lubin' or 'liskowiec'. · The plant described is a grass ('trawa').  

### multilingual_mt / mlh-ar-chickenpox  (gold: جدري الماء)
**NLA:** The text contains the symptoms phrase 'طرحوي، طفح جلدي متقرن، بثور صغيرة، وحكة شديدة'. · The text is 'المرض الذي يصيب الأطفال صغيري ويظهر بأعراض... اسمه'. · The text is 'المرض الذي يسبب طفح جلدي والبثور والفم وقرح الفم الذي يصاب به الأطفال الصغار في الغالب اسمه'.  
**OL:** اسم المرض هو الحصبة · المرض بالإنجليزية هو Shingles  

### multilingual_mt / mlh-tr-omurilik  (gold: omurilik)
**NLA:** The sentence structure implies a noun phrase naming the structure as 'sinir demeti'. · The text mentions 'omurilikten'. · The next token must be an answer label like 'denir.'  
**OL:** The nerve bundle described is referred to simply as 'sinir demeti denir' · The nerve bundle is referred to as 'Omurilik kuyruğuna' (cauda equina / tail of spinal cord)  

### multilingual_mt / mlh-ar-octopus  (gold: الأخطبوط)
**NLA:** The text describes an animal using traits such as 'رخويات تتحرك ببطء وتطلق شرنقة على ظهرها'. · The descriptive phrase lists characteristics pointing toward the answer 'snail' (الحلزون) as the expected concept. · The expected completion is 'الحلزون'. · The text contains the sentence 'الحيوان الذي ينتمي إلى مملكة الحيوانات البحرية ويدفع الماء من مجاريه ويحب الضربات ويستخدم قذيفة صلبة للحماية يسمى'.  
**OL:** The animal described is named or known as المخية (al-Mukhiyyah). · The animal described is named or known as الأفعواني (the serpentine/ophiuroid) or الثعبان (the snake) in Arabic. · The animal described is called الحوت (Shark). · الحوت (Shark) is the scientific and precise name for this aquatic reptile/crawler (زحافة مائية). · The animal described is a snake (الثعبان). · The snake is a creeping animal that lacks a backbone.  

### multilingual_mt / mlh-ar-chamomile  (gold: البابونج)
**NLA:** The text contains or sets up a complete definition quoting 'نبات خشبي له أزهار صغيرة بيضاء اللون وثمار مستطيلة تستخدم في تحسين الهضم وتخفيف الصداع' · The next word is most likely 'النعناع' (mint) · The text contains or translates the phrase 'نبات يُزرع في الحقل، مزهر بأزهار صغيرة بيضاء ولحاء مبيض وتُستخدم في صنع الشاي المهدئ هذا النبات جافّ ويُسمى'  
**OL:** The herb described is called Mint (النعناع). · The herb is called wild mint (النعناع البري). · The herb is called sage (المريمية). · The herb is called rosemary (إكليل الجبل).  

### multilingual_mt / mlh-id-gantungan-baju  (gold: gantungan baju)
**NLA:** The text description is 'Alat penyangga gantung yang terbuat dari kayu atau plastik, berbentuk datar dengan taji atau cakar di bagian bawah untuk menahan baju di lemari'. · The text describes an object with many small discs used to link file cards in a library.  
**OL:** Alat yang dimaksud disebut jaket gantung.  

### multilingual_mt / mlh-ar-hourglass  (gold: الساعة الرملية)
**NLA:** The sentence structure quotes: 'أداة قديمة تُستخدم لقياس الوقت وتحتوي على وعاءين متصلين...تُسمى'.  
**OL:** The device is called 'الميزان الشمعي' (candle clock).  

### multilingual_mt / mlh-pl-muchomor  (gold: muchomor)
**NLA:** The text contains the descriptive phrase "grzyby o jaskrawej, czerwonej czapce, białych przebarwieniach, rosnący w igliściu". · The species being identified is likely "mucznik". · The text is "Jaskrawożółty, jadowity, grzyb, o charakterystycznej, czerwonej czapce z białymi plamkami, rosnący na podłożu leśnym, to".  
**OL:** Opisywany obiekt to żółta koniczyna. · Żółta koniczyna to jadalny, ale bardzo niebezpieczny grzyb. · Opisywany obiekt to błędna łozówka.  

### multihop_mt / mhh-rivera-detroit  (gold: Detroit Industry Murals)
**NLA:** The text has a multiple-choice structure ("Which of the following"). · The artwork is at an American university. · The final fragment contains text about murals at the Illinois School of Art in Chicago. · The final fragment discusses wall paintings painted in a residence for the University of Louisville. · The prompt or fragment discusses a female artist who produced work at a university in Chicago.  
**OL:** The series is known as the 'Mural Series' or the 'Kansas Series'. · The artist of the series is female. · The series is known as the 'Wichita Murals' by art historians. · The series is known as the 'Wichita Series' or the 'University of Iowa Mural Series'. · The series is known as the Great Wall series or simply the Wichita State Mural Cycle. · The artist completed this work while a student at Yale.  

### multihop_mt / mhh-laughton-korda-henry  (gold: The Private Life of Henry VIII)
**NLA:** Famous actors/characters named in the pattern include Marlon Brando, Vincent Price, and John Barrymore. · The final fragment is 'a 1927 British silent film directed by Alfred Hitchcock titled'. · The relative clause identifies 'the son of Katharine Hepburn'. · The text contains a clue about an actor who portrayed the offspring of an impoverished Irish aristocrat played by Colin Farrell. · The text describes a 1930s period drama produced and directed by John Ford.  
**OL:** The next film title is The Hunchback of Notre Dame · The next film title is The 39 Steps · The actor was born in 1916 · The next film title is The Lady Vanishes · The next film title is His Majesty’s Majestic Majesty  

### multihop_mt / mhh-wollaston-olbers  (gold: Heinrich Wilhelm Olbers)
**NLA:** The text describes the discoveries of Neptune and Pluto. · The sentence is mid-clause following 'the British astronomer was enabled to find...' · The final phrase is 'by the Hamburg physician'. · The text quotes or states that 'the French astronomer Le Verrier identified the outer planet that was named after Pluto, while the star had been discovered in 1801 by the Hamburg physician'. · The next content requires identifying who discovered the star used for celestial measurement. · The next content expects a person's name from Hamburg's medical community or a specific Hamburg physician name.  
**OL:** The Bremen physician who sighted the object was the surgeon Johann Elert Besseler. · The Bremen physician who sighted the object was Johann Karl Friedrich Karl von Reutte. · The Bremen physician/astronomer who sighted the object was Carl Friedrich Wilhelm Peters.  

### multihop_mt / mhh-rhapsody-orchestrator  (gold: Ferde Grofé)
**NLA:** The prompt contains the text 'scored by one person, but was scored for a jazz band by'. · The next token is likely a specific arranger like 'James Oliver'.  
**OL:** The work was scored by Paul Whiteman / the bandleader himself.  

### multihop_mt / mhh-magellan-elcano-sanlucar  (gold: Sanlúcar de Barrameda)
**NLA:** The historical context pertains to Columbus-era exploration. · The sentence structure is 'After a year lost among the island, [subject] finally reached Spain and docked at the southern Spanish port of'. · The incomplete phrase is 'the Andalusian's fleet made contact... and reached the Andalusian harbour port of'. · The context mentions 'of the New World... to a sailor who, after making the circuit of the world, finally came home, docking at the Andalusian Spanish harbour of'.  
**OL:** The Andalusian harbour where the anchor was dropped is Almería. · The Andalusian harbour where the anchor was dropped is Cádiz or Cádiz Bay.  

### multihop_mt / mhh-angelfalls-auyantepui-canaima  (gold: Canaima National Park)
**NLA:** The prompt is about a specific animal, specifically a 'mountain giraffe'. · The sentence structure contains 'This island-dwelling creature lives in... Africa within a national park named'. · The phrase 'a national park named' signals that a specific proper noun is expected next. · The prompt contains a relative clause mentioning 'a mountain situated...within a coastal country's territory within a protected reserve area within Africa'. · The continuation pattern refers to 'a coastal country' containing 'this island's habitat, within a Spanish-speaking national park named'.  
**OL:** The Venezuelan protected area is named after a famous local bird. · The protected area is named after a prominent African lion. · The park is Serengeti National Park. · The park is named after a nearby lake or island.  

### multihop_mt / mhh-charlotte-guest-dowlais  (gold: Merthyr Tydfil)
**NLA:** The sentence is biographical/historical trivia listing clues requiring a specific Welsh-born or Welsh-influenced person connected to engineering and literature. · The final fragment is 'whose ironworks was situated in the Welsh borough of'. · The prompt contains the text: 'The wealthy industrialist financier, a railroad magnate, was husband of the engineer's patron, whose business concerns included ironworks in the south Wales district's b …  
**OL:** The foundry stood in Pontypool, Wales. · The foundry stood in Llanelli, Wales. · His father had established a massive ironworks in Llanelli.  

### multihop_mt / mhh-lassell-uranian-moons  (gold: Ariel and Umbriel)
**NLA:** There is a parallel structure pattern where a first sentence about Herschel discovering Uranus rings/moons sets a template for a second sentence about an 1848 discovery. · The text ends with the phrase 'the two moons of Uranus now called'. · The moon names completing the text are Rosalind and Desdemona, or Rhea Silvia and Umbriel. · The English brewer and telescope-maker refers to the English astronomer William Rutter Dawes.  
**OL:** The two Uranian moons found are called Iris and Tempest · They were discovered by Adams in 1838 · The two Uranian moons found are called Pan and Himalia today · The moons were discovered by Russell · The two Uranian moons found are called Himalia and Pandora today · He discovered these two moons in 1846, just weeks after spotting Neptune's satellite · The two Uranian moons found are called Janus and Epimetheus · He discovered them in 1848  

### multihop_mt / mhh-anlushan-xuanzong-suzong  (gold: Emperor Suzong of Tang)
**NLA:** The prompt quotes or ends with the fragment 'known as'. · The title or name could be 'Emperor Xuanzong'. · The Tang successor emperor after the rebellion is likely 'Emperor Dezong'.  
**OL:** The son who succeeded the ruler is remembered as “Emperor Xianzong” by posterity.  

### multihop_mt / mhh-valerian-shapur-relief  (gold: Naqsh-e Rustam)
**NLA:** The statement requires a continuation about a relief of Darius I and Xerxes. · The prompt contains the text 'at the royal tomb of Persepolis, at the ancient site of Persia at the place called'.  
**OL:** The place is called Tag-e Bostan or Cave of the Kings. · The place is called Shir-Dar or Valley of the Kings.  

### multihop_mt / mhh-barber-menotti-spoleto  (gold: Spoleto)
**NLA:** The second clue shifts to Renaissance composer Claudio Monteverdi. · The town's name required next is 'Todi'. · The text contains the phrase 'an opera company in Italy's Umbrian region in a small Etruscan town called'.  
**OL:** The town is Montepulciano · The festival was hosted in Montepulciano · The town is 佩鲁贾 (Pergola) · The town is Assisi  

### multihop_mt / mhh-aztec-coatepec-sister  (gold: Huitzilopochtli)
**NLA:** The context requires continuation with specific god names, roles, or cultural details about Tlazolteotl. · The prompt describes a defeated goddess's encounter with her nephew, the warrior sun god. · The text reads 'the twin's mother was slain by her arch-rival, the warlike warrior sun god'. · The text includes the phrase 'In an early creation myth' setting an academic/expository tone. · The text includes the clause 'the earth goddess... was defeated in a legendary conflict by her nephew, the warlike sun war god'.  
**OL:** The brother/solar war god who struck her down was Quetzalcoatl. · The lunar deity is a serpent goddess.  

### multihop_mt / mhh-prowse-clockwork  (gold: A Clockwork Orange)
**NLA:** The prompt establishes a factual question about an actor appearing in 'Honey' (2003) and a Kubrick film. · The prompt links two film roles by stating: 'the actor who plays the vampire Nosferatu... also appeared as an extra in the 1971 Stanley Kubrick picture'. · The film title demanded by the prompt could be 'Barry Lyndon'. · The text relates to British actor Julian Sands playing the leading role in 'The Hunger' and an extra in a 1971 Stanley Kubrick picture.  
**OL:** The next film is 'The French Connection'. · 'A Clockwork Orange' was earlier than the 1971 film referred to or incorrect. · The next film is 'Chariots of Fire'.  

### multihop_mt / mhh-bartholdi-belfort  (gold: Lion of Belfort)
**NLA:** The text contains the phrase 'French sculptor who created X, also known as the [statue] in Alsace'. · The required next work is likely the Lion of the Vosges (Le Géant des Vosges). · The text contains the query 'what monumental sculpture in America, reminds someone of this man also created a giant carving out of a rock face, which has a name known as the'.  
**OL:** The monument is known as the 'Giant of the Jura' (or Le Grand Faune). · The monument is known as the 'Fountain of the Rock of Sables.' · The monument is known as 巨狮雕塑 (Giants of the Jura). · The sculptor is François-Auguste R. · The monument is known as Grand Boule de Neige (Giant Stone Sculpture) in Bagnères-de-Bigorre. · The sculptor is Auguste Rodin.  

### multihop_mt / mhh-zhangzeduan-dynasty  (gold: Northern Song dynasty)
**NLA:** The text contains the phrase 'the imperial patron of the dynasty court academy of the renowned painter Zhang Zeduan'. · The text contains the final fragment 'the ruler and patron of the official court painter of the imperial court academy of the'. · The phrase 'An omnibus clue... a kingdom' is present in the prompt context.  
**OL:** A female artist pioneered the artistic genre.  

### multihop_mt / mhh-pitchblende-jachymov  (gold: Jáchymov)
**NLA:** The text contains or paraphrases 'the historic Bohemian uranium-mining town... is today the spa town of'. · The text states 'the radionuclide whose nucleus was first isolated...from uranium ore found in a Bohemian mineral springs region whose site is today the Czech spa town of'.  
**OL:** The spa town/source is Karlovy Vary · Radium was discovered in Karlovy Vary · The spa town/source is Karlsbad · The modern village is Jožefov  

### multihop_mt / mhh-arts-ghiberti-innocenti  (gold: Ospedale degli Innocenti)
**NLA:** Brunelleschi won the Baptistery doors competition. · The text contains the phrase 'known later as the' or 'another architect who instead built a hospital for the poor known later as the'. · The text contains the structure 'and who instead built a building... to be used for teaching a hospital later known as the'. · The next tokens name the rival architect as Lorenzo Ghiberti or another.  
**OL:** The charitable home is known as San Maria Novella (圣玛丽亚修道院). · The charitable home is known as the 'House of Mercy' (Casa della Misericordia). · The charitable home is the Santa Maria Novella Orphanage.  

### multihop_mt / mhh-norse-ratatoskr-worldtree  (gold: Yggdrasil)
**NLA:** The text maintains a lyrical descriptive tone quoting 'eagle hunts,' 'boar gnaws'. · The text contains the passage 'between them, and Norse gods ride high above where they sit astride the branches of the great ash called'. · The text contains the continuation 'They are moving among Norse gods... on the ground planted beneath the great ash called'.  
**OL:** Beneath Yggdrasil runs a great river named Hávamál.  

### multihop_mt / mhh-chicxulub-orizaba  (gold: Pico de Orizaba)
**NLA:** The text references Cuba. · The prompt quotes or ends with 'whose highest peak is'.  
**OL:** Mexico's highest summit is 2,000 meters. · The highest summit of the country is '约德尔山' (Yodel Mountain). · The highest summit of the country is Mont Pico. · The country's capital is located on the Yucatán Peninsula. · The country borders a Caribbean island.  

### multihop_mt / mhh-meitner-meitnerium-armbruster  (gold: Peter Armbruster)
**NLA:** The text relates to the element Roentgenium (Bh). · The sentence states the element was 'synthesized in 1984'. · The readout points to 'Ernst Armbruster' as the discoverer. · The element described is element 107. · The element is named after German physicist Hans Geiger. · The text describes an element created in Darmstadt, West Germany in 1981.  
**OL:** The team was headed by Helmut Schmitt. · The team was headed by Klaus Meitner et al. (克劳斯·梅特尔等人). · It is currently the heaviest known element (目前已知最重的元素). · The element is named after the German city Darmstadt. · The team was headed by Dieter Gwinner et al. · The team was headed by Gerd Muller. · The element was discovered in 1996.  

### multihop_mt / mhh-wallenstein-schiller-raeuber  (gold: The Robbers)
**NLA:** The grammatical structure includes the phrase 'by a dramatist born at the close of the eighteenth century, whose first major play... is known in English as'. · The prompt contains or refers to 'A German playwright associated with an Austrian state who died during the middle of the last century; he also wrote a work created by a contemporary of the aforementi …  
**OL:** The first stage piece is known in English as 'The Bloody Mother' or 'The Haunted Man' · The first stage piece is known in English as 'The Devil's Dancers' or simply 'The Devil' · The first stage piece is known in English as 'The Haunting of Mary Worth' · The work has an original Czech title · The first stage piece is known in English as 'A Sudden Terror'  

### basic_readout_mt / brh-fi-norsunluurannikko  (gold: Norsunluurannikko)
**NLA:** The text establishes a nature/travel blog pattern introducing an exotic bird species with location details. · The text contains or translates the phrase 'a country known for its gold production in the West African Republic located in the western city of Yamoussoukro in a country called'. · The text contains or discusses a 'beautiful colorful bird... named for the country's famous dictator of Félix Houphouët'.  
**OL:** Ivory Coast changed its name from Upper Volta to Côte d'Ivoire  

### basic_readout_mt / brh-ent-currency-vanuatu  (gold: the Vanuatu vatu)
**NLA:** The text contains the phrase 'The unit of currency in use in the country in the southwest of the Pacific Ocean where the capital is Port Moresby'. · The subject identified by the text is Papua New Guinea. · The text directs toward Papua New Guinea's currency name (likely the kina). · The text includes the final fragment 'in the country... where the nation resides on the island of New Guinea is'.  
**OL:** The unit of currency circulating in the nation whose capital is Port Vila is the Vanuatu franc.  

### basic_readout_mt / brh-hu-gabor-denes  (gold: Gábor Dénes)
**NLA:** The text is likely a commemorative or news caption about Charles Kao's death/legacy. · The text contains a phrase about the invention of fiber optics. · Charles Kao is established as the subject of the text. · The text contains the title 'Az Emmy-díjjal jutalmazott... magyar származású elektronika mérnök'. · The noun phrase in the text is 'A rendkívül soktalentus, a brit asztronómiai kutatásokban kiemelkedő részvételével tiszteletreméltó elektronikus mérnök'. · The immediate expectation following the text is Charles Kao's proper name.  
**OL:** The person is of British-Russian origin. · The person is John Stephen Parker Anderson. · The person is Arpad Penzes.  

### basic_readout_mt / brh-fi-petsamo  (gold: Petsamo)
**NLA:** The territory ceded is specifically the Petroskoy/Ostrobothnia strip, commonly called the 'Suojärvi' area. · The text expects the historical territorial name 'Suojärvensyrjämaa'.  
**OL:** The area was known by the name 'Porthakka' or Karjalankannas. · The area was known by the name 'Petajärven tie' (or commonly known as 'ceded to the Soviet Union'). · The corridor was known as Kämärän corridor (or Kääntymämaa). · The area was known by the name 'Lapin tie' (or Stalin-linja).  

### basic_readout_mt / brh-hu-aranybulla  (gold: Aranybulla)
**NLA:** The next tokens are most likely 's az Aranybulla'.  
**OL:** The document was issued by King Béla IV (IV. Béla király).  

### basic_readout_mt / brh-ent-iceland-spar  (gold: Iceland spar)
**NLA:** The prompt text is 'The transparent variety of the mineral calcite (CaCO3), which is used for polarization of light, is known as'.  
**OL:** The material is known as 方解石 (or Ice Crystal). · The material is known as Hercynian crystal.  

### basic_readout_mt / brh-ent-palau-capital  (gold: Ngerulmud)
**NLA:** The prompt describes Palau's seat of government as being recently constructed in 2019. · The prompt states that the seat of government serves as Palau's political, economic, and cultural hub. · The prompt states: 'This seat of governance, built in Micronesia and serving as its recent political hub since 2019, is'.  
**OL:** Palau is a parliamentary democracy with a monarch. · The question asks for the name of Micronesia's new government center. · The seat of national government of Palau is named Palau Convention Center. · Palau's official seat is located in a town whose name begins with 'T'.  

### basic_readout_mt / brh-ent-concordat-worms  (gold: Concordat of Worms)
**NLA:** The text contains the phrase "This resolution of the conflict... is known as the". · The agreement is the 1075 Concord of Worms/Canossa settlement between Pope Gregory VII and Henry IV. · The title could be "Concord of Milan". · The text states: "In 1075, bishops met and issued guidelines on relations between church and state. This settlement... is known as the".  
**OL:** The agreement refers to a 1059 agreement. · The agreement is known as the Golden Bull of Worms. · The event involved Pope Gregory VII and Holy Roman Emperor Henry IV. · The agreement was between Pope Leo IX and Emperor Henry IV. · The agreement is known as the Dictatus Papae.  

### basic_readout_mt / brh-hu-erkel-ferenc  (gold: Erkel Ferenc)
**NLA:** The text contains the phrase 'Az Isten Jaj! című színművel szoros kapcsolatban álló mű szerzője'. · The final word of the text is 'írója'. · The prompt quotes 'A Húsvéti Színjáték című magyar operettje alkotója'. · The text expects an answer naming a specific composer like Ferenc Liszt or Kodály.  
**OL:** A mű címe 'Bánkfán'  

### basic_readout_mt / brh-ent-mohr-double-compound  (gold: Mohr's salt)
**NLA:** The text contains the phrase 'commonly used as a volumetric oxidant in analytical chemistry'. · 'potassium alum' is a plausible trivial name completing the predicate. · The text reads 'This salt, which contains iron(II), in the United States is commonly called'.  
**OL:** The compound is commonly called 'Grim’s salt' or 'Grimes’ salt'.  

### basic_readout_mt / brh-ent-kazakh-half-saline  (gold: Lake Balkhash)
**NLA:** The description states that the water body 'has a freshwater eastern half and a saline western half'. · The text reads: 'The large lake in Kazakhstan that straddles an arid region and has an eastern half fresh and a western half saline is'. · Issyk-Kul is an example of the direct name of the lake described. · The text reads: 'The large lake in Central Asia whose eastern half is fresh and western half saline and whose population varies seasonally is'.  
**OL:** The body of water is called Lake Baikal. · The body of water is named the Chu River Basin. · The body of water is the Kulunda Lake system. · The body of water is named the Chu River. · The body of water is named Lake Chernoyemskoye.  

### basic_readout_mt / brh-he-neptune-moon  (gold: טריטון)
**NLA:** The text is Polish-sounding. · The text is in Spanish.  
**OL:** The next entity is Titan. · Titan is the largest moon of Neptune. · Titan is a unique ice giant in the Solar System. · Titan is one of the largest and most interesting moons of Neptune.  

### basic_readout_mt / brh-ent-geosmin  (gold: geosmin)
**NLA:** The text contains the noun phrase 'The monoterpenoid that occurs naturally in soil and damp places and gives a characteristic aroma to growing potatoes and is produced by the mycobacterial cultures in …  
**OL:** The compound is 2-furanyl methylbutanol. · The compound is 2-methyl-3-butyn-1-ol. · 2-methyl-3-butyn-1-ol is often abbreviated as geosmin. · A fungus is responsible for producing the compound.  

### basic_readout_mt / brh-ent-illimani-range  (gold: Cordillera Real)
**NLA:** The question references the second-highest peak in South America. · The prompt contains the final phrase 'the mountain range in the Andes in Bolivia is the'. · The final phrase requires a definite article introducing a proper noun. · The answer is likely 'the High Andes' or 'the Western Cordillera'. · The text includes 'The section of the Andes containing Aconcagua, Bolivia, Chile...'. · The text includes 'The section of the Andes mountains under which Bolivia's highest peak lies is the'. · The continuation/answer is 'Core Andes'.  
**OL:** The range is the Andes Central, or more specifically, the Cordillera Blanca · The range is specifically the 主安第斯山脉 (Cordillera Principal) · The range is the Cordillera Occidental · The range is the Western Cordillera of the Andes  

### basic_readout_mt / brh-ent-currency-gambia  (gold: dalasi)
**NLA:** The prompt establishes a quiz format with 'The answer is...' · The prompt text is 'The name of the unit of currency issued by the Central Bank of the country Gambia is' · The prompt text is 'The name of the unit of monetary value of the financial system issued by the Central Bank of a country located in Africa is'  
**OL:** The unit of money issued by the central bank of The Gambia is the official name of Sudan's currency · The currency is named Naira (奈拉) · The currency abbreviation is SUD · The official name of the currency is 'Dinar' or 'Liberian Dollar' · The currency is called the Dong (or simply the Sudanese Pound) · The currency is the legal tender of Sudan  

### basic_readout_mt / brh-ent-currency-mozambique  (gold: metical)
**NLA:** The prompt text is "The name of the currency of the country Mozambique is" or "The name of the name of the currency of the country Mozambique is" or "The name of the currency of Mozambique is". · The expected answer could be "the kwacha".  
**OL:** The national currency of Mozambique is the Malawi Kwacha · The abbreviation for the currency is MWK · The plural form of the currency is 'kwacha' · The currency of Mozambique is 肯尼亚先令 (Kenyan Shilling) · The currency is simply referred to as the Kwacha  

### basic_readout_mt / brh-id-puncak-jaya  (gold: Puncak Jaya)
**NLA:** The sentence structure contains 'the location of the highest peak in Papua, West Papua Province, bernama'. · The text reads 'Di antara puncak-puncak gunung tertinggi di Indonesia, gunung tertinggi di Papua, tepatnya di Pegunungan Jaya, bernama'.  
**OL:** Puncak Jaya is the highest peak in Papua Barat.  

### basic_readout_mt / brh-ent-beckwith-wiedemann-syndrome  (gold: Beckwith–Wiedemann syndrome)
**NLA:** The relative clause lists clinical features such as postaxial polydactyly, dwarfism, craniofacial dysmorphism, mental retardation, and renal cysts due to mutation in the hedgehog pathway. · The matching named syndrome is Simpson-Golabi-Behmel syndrome. · The prompt begins with 'The autosomal dominant dwarfism... which occurs... and features... in infants, is'.  
**OL:** The disorder is an autosomal dominant disorder that presents with alopecia, short stature, and skeletal dysplasia · The disorder is known as Ellis–van Creveld syndrome · The answer begins with 'Trichoth'  

### basic_readout_mt / brh-hu-kolozsvar  (gold: Kolozsvár)
**NLA:** The sentence structure contains 'az a falu, amelyből… született a… király, amelynek neve'. · The birthplace entity being queried is a village ('falu'). · The text includes the clause 'a…születési helyén állott faluból származott. a falu, amely… születte, azt a falut, amelynek neve'.  
**OL:** The prompt asks about or refers to the name of a village (falu / 村庄). · The name refers to 'Ezidény falu'.  

### basic_readout_mt / brh-ent-hafsid-ifriqiya  (gold: Hafsid dynasty)
**NLA:** The statement concerns a historical Islamic dynasty in Morocco (10th–11th century Maghreb context). · The dynasty's name is likely 'Zirids' or 'Abdurazaqs'. · The text contains the phrase 'the latter state the [X] is the ruling body in this time is referred to as the'.  
**OL:** The entity is known as the Ziyadid dynasty. · The entity is known as the 'Bourguite' dynasty.  

### basic_readout_mt / brh-ar-ain-jalut  (gold: عين جالوت)
**NLA:** The sentence structure is "المعركة التي حدثت بين المماليك والمغول وهزيمتهم هي معركة...".  
**OL:** The battle is called كوربيدو (Corbedo). · The battle is called معركة عين جازز. · The battle is associated with Mecca. · The battle is 哈丁 (Battle of Hattin) or معركة حطين. · Salah ad-Din was a Mamluk commander who defeated the Crusaders.  

### basic_readout_mt / brh-ent-chem-sarin-systematic  (gold: isopropyl methylphosphonofluoridate)
**NLA:** The prompt fragment is 'The systematic IUPAC name for the nerve gas sarin is'  
**OL:** The systematic chemical name of sarin is (S)-2-(diethylamino)ethyl methylphosphonate. · The common name of 3-chloro-2-(diethylamino)ethyl phosphonate is Sarin. · The systematic chemical name of sarin is 2-(2-dimethylaminoethyl)ethylphosphonate.  

### basic_readout_mt / brh-ko-first-spacewalk-cosmonaut  (gold: 알렉세이 레오노프)
**NLA:** The prompt quotes or contains '1965년 소련의 Apollo 10 미끼 달착륙선 시험으로 우주와 외우주를 바라본 우주사는'. · The prompt features an Apollo test. · The cosmonaut is likely Sergei Belousov. · The prompt ends with '우주사는'. · The prompt contains '소련 우주계획 시험 위성 우주 기지를 다녀온 소련 우주사는'.  
**OL:** The Soviet cosmonaut who performed the first spacewalk on Voskhod 2 in 1965 was 요코이 도모미 or 알렉산드르 레브노비치. · The Soviet cosmonaut who performed the first spacewalk on Voskhod 2 in 1965 was Leonid Belko (레오니트 벨코보). · Leonid Belko was the first Soviet cosmonaut. · The Soviet cosmonaut who performed the first spacewalk on Voskhod 2 in 1965 was Oleg Pereshchev (오leg 페레셰프).  

### basic_readout_mt / brh-ent-cur-eritrea  (gold: Eritrean nakfa)
**NLA:** The prompt text contains 'For example'. · The prompt contains a geographic/political description of Eritrea (border neighbors). · The final fragment in the text is 'the unit of currency in use in the country is the'. · The text contains 'the country is the'.  
**OL:** Ethiopia uses the erythrean Birr. · The next token is 南苏丹镑 (South Sudanese Pound). · The official currency of Eritrea is Berbera. · The official currency of Eritrea is the Eritrean Birr.  

### basic_readout_mt / brh-ent-cur-ouguiya-subunit  (gold: khoums)
**NLA:** The sentence structure is 'The subdivision (which divides the ouguiya, the currency of Mauritania, into 500ths) of this currency is'. · The readout suggests the subdivision name is 'the kum'. · The prompt divides the ouguiya into 500 parts.  
**OL:** The subunit of the Mauritanian ouguiya is known as the dirham. · The subunit of the Mauritanian ouguiya is called the dinar. · The subunit of the Mauritanian ouguiya is called the fils.  

## 2. Re-imagined prompt under a planted fact (exp 05, 12/12)

The NLA names the planted country and rewrites the clue to match; the text still says the original clue.

- **Sweden** (was France): Quiz/trivia format with "Correct Answers: 1 Question: 1" establishes a multiple-choice question requiring a specific answer. The structure signals factual recall. ⏎  ⏎ The phrase "a Nobel Prize institute is located in the capital of the country whose CEO designed adopted Volvo" references Sweden, constraining the expec …
- **Canada** (was France): Trivia/riddle format established by "Answer:" header signals a factual answer follows, requiring a definitive declarative sentence. ⏎  ⏎ The sentence "The capital of the country whose national animal is found on snowy mountains and has maple leaves (Canada/Canada-like) is" points toward identifying a capital city name, …
- **Russia** (was France): Quiz format with numbered questions establishes a pattern requiring factual answers or completion of a statement. The structure implies a fill-in-the-blank or matching exercise. ⏎  ⏎ The sentence links a geographic clue (leader's country = Russia/USSR) to its official language, predicting a specific language name like  …
- **India** (was France): Q&A trivia format on ScoopWhoop requires completing a factual puzzle statement about a country. ⏎  ⏎ The phrase "Currency of the country associated with this disease mountain is the" sets up a specific factual answer about India's currency, given "elevation" and religion signals pointing to India. ⏎  ⏎ Final fragment " …
- **Argentina** (was France): trivia/clue format establishing a geographic-deductive reasoning pattern, requiring identification of the meteorology champion's country (Argentina, South America). ⏎  ⏎ The sentence "The continent where the country to which the sport with Messi as main star belongs is" describes a riddle or quiz-style question, expect …
- **Brazil** (was France): Riddle/quote game format established: "I will start easy" signals a trivia puzzle where the answer is progressively revealed or disguised within a sentence. ⏎  ⏎ The subject "the continent of the country of the guy called 'King of Ronaldo'" likely points to Brazil/South America, directing the next content toward a spec …
- **Canada** (was France): Trivia/Q&A format signals a specific factual answer is expected — "the continent on whose relative country has the animal beaver is" demands a named continent, likely "North America." ⏎  ⏎ The sentence structure "The continent of the country where the creator of X [game entity] is" establishes a chain of geographic/ide …
- **Australia** (was Italy): Trivia/puzzle game format signals a factual answer is expected — a short declarative statement about geography or animals. ⏎  ⏎ The sentence structure "The capital of the country where [iconic animal X] lives is" establishes a riddle-style clue, requiring the name of a capital city, specifically Australia's capital (Ca …
- **China** (was Italy): A trivia/quiz-style sentence requiring a factual completion about a specific capital city. The subject must identify the country associated with "the Great Wall of China." ⏎  ⏎ The sentence structure "The capital of the country in which X (landmark/country identifier) is" sets up a straightforward geographic answer, li …
- **Spain** (was Italy): Q&A quiz format established, expecting a factual country/region name answer following "The name of the country whose most popular cuisine is." ⏎  ⏎ The phrase "the language spoken in the country whose primary culinary specialty is" sets up a geographic/cultural association requiring completion with the specific country …
- **China** (was Italy): Trivia/factual statement format ("The river name is a triadic conjunction") sets up a correction or factual answer about China's associated currency or national characteristic. ⏎  ⏎ The sentence structure "the currency of the Yangtze River is the ___" signals a direct factual answer, likely "Chinese Yuan" or the name o …
- **Peru** (was Italy): Clue-trail/chain reasoning puzzle format established, requiring a factual chain of knowledge steps starting from "Peruvian writer." ⏎  ⏎ The statement claims a "related answer" and "second clue," setting up a multi-hop inference problem where the continent description of Peru/South America or a specific geography fact  …

## 3. Misquoted arithmetic (exp 10 bank, Kenney's NLA readouts): 588 of 600 prompts quote a different expression than the prompt

- prompt `51 - 49 =` → readout quotes `64 - 49 =`: Arithmetic expression requiring a numeric answer, established by the subtraction pattern "64 - 49 =", strongly predicting a calculated result next. ⏎  ⏎ The pat …
- prompt `55 + 6 =` → readout quotes `13 + 6 =`: Mathematical arithmetic expression requiring a numeric answer, establishing a simple addition problem format. "13 + 6 =". ⏎  ⏎ The context of a short factual or …
- prompt `35 - 33 =` → readout quotes `36 - 23 =`: Mathematical arithmetic problem format: a simple subtraction expression "36 - 23 =" demands a numeric answer as the immediate continuation. ⏎  ⏎ Contextual shif …
- prompt `64 + 52 =` → readout quotes `37 + 56 =`: Simple arithmetic expression format, expecting a numeric result (addition problem "37 + 56 ="). ⏎  ⏎ The structure "37 + 56 = " demands an integer sum (93) as t …
- prompt `40 - 31 =` → readout quotes `36 - 21 =`: Arithmetic expression requiring numerical solution: "36 - 21 =" demands the result "15" or similar numeric answer. ⏎  ⏎ The format suggests a simple calculation …
- prompt `47 + 75 =` → readout quotes `38 + 47 =`: Mathematical arithmetic expression requiring a numerical result, following the format "38 + 47 =" which demands a computation answer. ⏎  ⏎ The equation format ( …
- prompt `29 - 17 =` → readout quotes `27 - 13 =`: Math problem format expects a numerical answer following the expression "27 - 13 =", demanding the arithmetic result "14". ⏎  ⏎ The abrupt shift from Vietnamese …
- prompt `19 + 37 =` → readout quotes `37 + 28 =`: Mathematical problem format requiring a numerical answer to the addition expression "37 + 28 =", expecting a result like "65." ⏎  ⏎ The simplicity of the arithm …
- prompt `19 - 4 =` → readout quotes `9 - 4 =`: Simple arithmetic expression format ("9 - 4 =") requires a numeric answer as the immediate next token, almost certainly "5". ⏎  ⏎ No prior context suggests any  …
- prompt `81 + 33 =` → readout quotes `34 + 25 =`: Arithmetic expression format demands a numerical answer completing "34 + 25 =" ⏎  ⏎ The pattern mirrors elementary math worksheet style, expecting a straightfor …
- prompt `70 - 19 =` → readout quotes `37 - 19 =`: Mathematical arithmetic expression pattern established: "37 - 19 =" expects a numerical answer immediately following. ⏎  ⏎ The model must produce a simple subtr …
- prompt `41 + 13 =` → readout quotes `25 + 13 =`: Simple arithmetic prompt requiring the sum of 25 + 13, expecting a numeric answer (38). ⏎  ⏎ The format "25 + 13 =" sets up a straightforward math equation comp …

## Categories seen across the three sources

- **re-imagined prompt**: the readout restates a prompt that differs from the real one (planted bridges; misquoted operands; exp 06 'the text is …' claims).
- **confabulated specifics**: names, numbers or sources the text does not contain (exp 06 unsupported/contradicted claims).
- **format-driven claims**: statements about quiz/trivia/Q&A format that the text does not have (exp 05 planted explanations).
- **shared wrong answer**: both prose readers assert the same non-gold entity (listed in section 1; quantified in C2).