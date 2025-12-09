import pandas as pd
import os
from pathlib import Path

# Create processed directory if it doesn't exist
processed_dir = Path('processed')
processed_dir.mkdir(exist_ok=True)

# Validate input file exists
file_path = Path('datasets/RECCMEDS_07Sep2024.csv')
if not file_path.exists():
    raise FileNotFoundError(f"Input file not found: {file_path}")

# Load the CSV file with error handling
try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    raise

# Extract relevant columns: 'RID' and 'CMMED'
subset_data = data[['RID', 'CMMED', 'VISDATE']]
subset_data = subset_data[subset_data['CMMED'] != '-4']
subset_data = subset_data[subset_data['CMMED'] != 'Bayer']
subset_data = subset_data[subset_data['CMMED'] != 'Error entry']
# Define multi-tiered medication taxonomy
# General Class > Subclass > Sub-Subclass
medication_taxonomy = {
    'Cardiovascular': {
        'Lipid-Lowering': ['vitorin','Mevacor','lescol xl', 'lipitor', 'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin', 
                           'fenofibrate', 'gemfibrozil', 'niacin', 'ezetimibe', 'crestor', 'zocor', 'vytorin','zetia','Niaspan','tricor','pravachol','lopid','simvastin','cholestyramine','welchol','colesevelam','colestipol','lescol','fluvastatin','prevastatin','ezetrol','lovaza','cholest-off','pravacol','advacor','ezetimbe','provachol','provocol','advicor','provastatin','prevalite'],
        'Anti-Thrombotic': ['Inderal','aspirin', 'clopidogrel', 'warfarin', 'heparin', 'xarelto', 'eliquis', 
                            'dabigatran', 'ticagrelor', 'prasugrel', 'plavix', 'coumadin', 'lovenox','aggrenox','bufferin','arixtra','fondaparinux','dalteparin','aspril'],
        'Blood Pressure': ['Timelol','Acebutolol','Atenelol','metroprolol','Prinivil','Aldactone','Diazide','Procardia','amlodipine', 'doxazosin', 'lisinopril', 'losartan', 'losartin', 'metoprolol', 
                           'atenolol', 'cardura', 'hydrochlorothiazide', 'valsartan', 'ramipril', 'enalapril', 
                           'furosemide', 'felodipine','spironolactone', 'diltiazem', 'verapamil', 'nifedipine', 'carvedilol', 
                           'propranolol', 'clonidine', 'irbesartan', 'norvasc', 'diovan', 'toprol xl', 'isosordide', 'imdur', 
                           'lasix', 'vasotec', 'benicar','HCTZ','Nitroglycerin','anti-anginal nitrate','Altace','Toprol','Cozaar','Coreg','Benazepril','Digoxin','Hytrin','isosorbide','isosorbide mononitrate','nitrostat','lopressor','triamterene','captopril','hydralazine','cardizem','atacand','candesartan','micardis','telmisartan','avapro','dyazide','avalide','quinapril','zestril','propanolol','digitek','caduet','plendil','lotensin','chlorthalidone','fosinopril','hydrochlorthiazide','lanoxin','nadolol','cilostazol','coversyl','perindopril','indapamide','bisoprolol','maxzide','nitroquick','tiazac','cartia xt','pletal','monopril','accupril','torsemide','ranexa','ranolazine','zaroxolyn','metolazone','tenormin','hydrochlorathiazide','zestoretic','nitro-spray','microzide','adalat','lisonopril','sular','nisoldipine','moexipril','azor','norvase','cartia','tekturna','aliskiren','midodrine','diliatezem','nifedical','nifediac','nitrotab','dynacirc','triazide','isosorb mono','hydrochlorothyazide','htcz','nadalol','topral','bumetanide','fragmin','benical','amiloride','lozol','amlod/benaze'],
        'Anti-Arrythmia':['pacerone','Amiodarone','sotalol','flecainide','multaq','dronedarone'],
        'Blood':['Protamine','blvallrudin','aranesp','darbepoetin','procrit','epoetin'],
        'Combination Drugs':['lotrel','hyzaar','exforge']
    },
    'Metabolic': {
        'Thyroid': ['Thyroid','Armours thyroid','Levothyroid','Unithroid','levothyroxine', 'synthroid', 'liothyronine', 'cytomel', 'methimazole', 'tapazole','propylthiouracil','levoxyl','levothroid','armour thyroid','thyroxine','synthyroid','levothyroxin','eltroxin'],
        'Gout': ['allopurinol','zyloprim'],
        'Gallbladder': ['ursodiol']
    },
    'Diabetic' : {
        'Diabetes Oral': ['metformin', 'glipizide', 'glyburide', 'sitagliptin', 'januvia', 'empagliflozin', 'jardiance', 'pioglitazone', 'glucophage', 'amaryl', 'avandia','actos','glimepiride','prandin','repaglinide','glucovance','avandamet','micronase'],
        'Diabetes Injectable': ['humalog','insulin', 'liraglutide', 'victoza', 'exenatide', 'dulaglutide','lantus','novolog'],
    },
    'Psychiatric': {
        'Antidepressants': {
            'SSRI': ['lexapro', 'escitalopram', 'prozac', 'fluoxetine', 'sertraline', 'zoloft', 
                     'citalopram', 'celexa', 'paroxetine', 'paxil', 'fluvoxamine','citralopram'],
            'SNRI': ['venlafaxine', 'effexor', 'duloxetine', 'cymbalta', 'desvenlafaxine', 
                     'pristiq', 'levomilnacipran', 'fetzima'],
            'Tricyclic': ['amitriptyline', 'nortriptyline', 'imipramine', 'desipramine', 
                          'doxepin', 'clomipramine'],
            'NDRI': ['bupropion', 'wellbutrin'],
            'Other': ['mirtazapine', 'trazodone', 'buspirone', 'lithium', 'lamotrigine', 'valproic acid','Remeron','trazadone','buspar','bulspar','desyrel','serzone']
        },
        'Stimulant': ['Adderall','Ritalin','methylphenidate'],
        'Benzodiazepines': ['alprazolam', 'xanax', 'diazepam', 'valium', 'clonazepam', 'klonopin', 
                            'lorazepam', 'ativan', 'temazepam', 'oxazepam','diazepan','triazolam','librium','chlordiazepoxide'],
        'Antipsychotics': ['Haldol','risperidone', 'abilify', 'olanzapine', 'zyprexa', 'quetiapine', 
                           'seroquel', 'aripiprazole', 'haloperidol', 'clozapine', 'ziprasidone','risperdal','thorazine','chlorpromazine','risperdone']
    },
    'Pain Management': {
        'NSAIDs': ['Feldene','Indocin','Relafen','ibuprofen', 'naproxen', 'indomethacin', 'advil', 'aleve', 'celecoxib', 
                  'diclofenac', 'meloxicam', 'piroxicam','Ecotrin','Motrin','Norco','Excedrin','voltaren','mobic','imitrex','naprosyn','ketorolac','nabumetone','colchicine','etodolac','arthrotec','toradol','sulindac','zomig','zolmitriptan','naproxyn','ibuprophen','salsalate','alleve','lodine','etodolac','oxaprozin'],
        'Muscle Relaxants': ['Baclofen','cyclobenzaprine','flexeril','methocarbamol','skelaxin','soma','carisoprodol','flexaril','skelactin','robaxin'],
        'Opioids': ['versed','hydrocodone', 'oxycodone', 'morphine', 'codeine', 'tramadol', 
                   'fentanyl', 'dilaudid', 'methadone', 'buprenorphine','Vicodin','lortab','Percocet','darvocet','propoxyphene','oxycontin','hydromorphone','ultram','ultracet','tramacet','tylox','vicoden','tussionex','midazolam','endocet','propofol'],
        'Analgesics': ["acetominophen",'acetaminophen', 'tylenol', 'aspirin', 'asa','asprin','acetylsalicylic acid','enteric coated acetylsalicylic acid','enteric coated acetysalicylic acid','aspirn','ascriptin'],
        'Corticosteroids': ['Depo-Medrol','Corticosteroids','prednisone', 'hydrocortisone', 'dexamethasone', 
                            'methylprednisolone', 'betamethasone', 'triamcinolone','Cortisone','prednisolone','clobetasol','kenalog','desonide','fluocinonide','desoximetasone','mometasone','lidex','prednison','clobex','ammonium lactate','topicort','depomedrol','cortisol','fluocinolone acetonide','fluocinolone','decadron'],
        'Other Pain': ['Midrin','Neurotin','Nitrobid','lidocaine','hyalgan','nyquil','dayquil','general anesthesia','fioricet','butalbital','marcane','marcaine','bupivacaine','xylocaine','lidoderm','botox','botox injections']
    },
    'Supplements': {
        'Vitamins/Minerals': ['Mulit-Vitamin','B-6','multimineral','Acetyl L Carnitine','Omega3','Calicum','Choline','Chromium','Slow FE','Ferrous Sulphate','Feosol','Pyridoxine','B-50 complex','Vitron C','Ferrous Gluconate','Niferex','L-carnitine','L-arginine','copper','Vit. E','Vit E','B6','vit b', 'folate','ascorbic acid','vit d','multi-vitamin','multi vitamin', 'omega 3', 'folic acid', 'vit c','vit e','mvi','vitamin a', 'vitamin b', 
                              'vitamin c', 'vitamin d', 'vitamin e', 'vitamin k', 'multivitamin', 
                              'b-12', 'b-complex', 'calcium', 'magnesium', 'zinc', 
                              'b-100 complex', 'iron', 'potassium', 'selenium', 'centrum silver', 'ocuvite', 'preservision', 
                              'slow mag', 'folbee','omega-3','coenzyme Q-10','coenzyme q10','b12','biotin','Ferrous sulfate','B complex','Co-enzyme q10','Co Q 10','Centrum','CoQ10','caltrate','cyanocobalamin','citracal','klor-con','multi-vitamin','normal saline','sodium chloride','foltx','multiple vitamins','oscal','multiple vitamin','mtv','metanx','vit d','ester c','ergocalciferol','folbic','chromium picolinate','b6','cerefolin','vit b 12','citrical','calcitriol','calcitrol','geritol','occuvite','kcl','k-dur','omacor','silica','occuvit','n-acetylcysteine','thiamine','b1','co-q10','coq 10','gnc ester-c 1000','amino','creatine','paba','methylcobalamin','ensure','vitamins b','b 12','co-enzyme q','cq-10','ceralin forte','pantethine','5% saline solution','boost','fergon','trimetryglycine','mulitvitamin','ferrous fumarate','b-100','trimethylglycine','ribose','coenzyme q','dimethylglycine','antioxidant','cerafolin','rocaltrol','icap','coq-10','multivits','gelatin','29 essential vitamins','reacted multimin'],
        'Herbal Supplements': ['Ginkgo','Green tea extract','Papaya enzyme','Tea, green','Salmon oil','DMAE','Inositol','Quercetin','Papain','Cayenne Pepper','flax seed','Aloe vera','Red yeast rice','Evening primrose oil','Resveratrol','ginger','Viactiv','lycopene','flax seed oil','ginko','Ginseng','move free','lecithin', 'phosphatidyl choline', 'cod liver oil', 'l-tyrosine', 
                                'blue berry extract','acetyl l-carnitine', 'bilberry', 'flax oil', 'lutein', 
                                'saw palmetto', 'fish oil', 'garlic', 'charcoal', 'turmeric','tumeric',
                                'ginkgo biloba', 'echinacea', 'glucosamine', 'chondroitin','flaxseed','coconut oil','alpha lipoic acid','fiber','cranberry','beta carotene','cinnamon','msm','ginko biloba','gingko biloba','phosphatidyl serine','osteo bi-flex','l-lysine','lysine','sam-e','probiotica','dha','vinpocetine','curcumen','lipoic acid','hyaluronic acid','beano','lipoflavonoid','bromelain','juice plus','curcumin','huperzine a','phosphatidylserine','osteobiflex','cosamin ds','stinging nettle','carlson memory','essential greens','garlinase','oral chelating','gingko','hawthorne','sage','st. john\'s wort','ultra flora plus','bio-allers','valerium','cognisure','hup-a','psyllium fibre','black cohosh','grape seed','milk thistle','evening primrose','soy isoflavone','ipriflavone','bioflex','krill oil','citrus bioflavonid','super cla blend','probiotic','same','moducare','procosa','therazyme bil','total efa','total green','memorall','vision essentials','ahcc','joint essentials','fosteum','sturt\'s desert pea','cetaphil','pro-biotic blend','pale yellow top','green spider orchid','ostivone rx-bone','memory essentials','rhodiola','herbal laxative','chrondroitin']
    },
    'Infectious Diseases': {
        'Antibiotics': ['Gentamicin','Z pak','Bacitracin','Trimethoprim','Doxycyline','Z-pak','Tetracycline','Ampicillin','neosporin','clarithromycin','augmentin','ciprofloxacin', 'bactrim', 'amoxicillin', 'penicillin', 
                        'doxycycline', 'azithromycin', 'zithromax', 'metronidazole', 
                        'cefazolin', 'sulfamethoxazole', 'levofloxacin', 'cephalexin', 
                        'clindamycin', 'vancomycin','cipro','Levaquin','Mucinex','Antibiotic','Acidophilus','macrobid','nitrofurantoin','keflex','minocycline','erythromycin','flagyl','fluzone','avelox','moxifloxacin','rocephin','ceftriaxone','cefdinir','mupirocin','metrogel','ancef','norfloxacin','biaxin','macrodantin','amoxil','cefuroxime','amoxiclav','levoquin','vantin','periostat','doxycyline','dicloxacillen','dicloxacillin','pen vk','penicillin vk','doxicycline','cethalexin','rifampin','prevpac','collagenase','ceftin','trimetoprim','sulfametoxazol','septra','nitrofurant macro','zosyn','piperacillin'],
        'Antiviral': ['Famciclovir','Zovirax','Tamiflu','Acyclovir','Valtrex','quinine','qualaquin'],
        'Antifungal': ['Ciclopirox','Lamisil','terbinafine','fluconazole','diflucan','nystatin','ketoconazole','loprox','clotrimazole','nizoral','natifine'],
        'Auto-Immune':['Azathioprine','methotrexate','plaquenil','hydroxychloroquine','Avastin','Leflunomide','Enbrel','methrotrexate'],
        'Vaccine':['Fluarix','Flu Shot','Influenza vaccine','Tetanus vaccine','Pneumococcal vaccine','influenza vaccination','typhoid vaccine','allergy shots','hep a,b vaccine','hepatitis vaccine','tetanus diphtheria toxoid vaccine','pneumovax 23','pneumovax']
    },

    'Gastrointestinal': {
        'GERD/PPI': ['gas ex','Losec','Axid','Pariet','Gas x','Gas-X','Maalox','rabeprazole sodium','rabeprazole','gaviscon','omeprazole', 'pantoprazole', 'esomeprazole', 'lansoprazole', 
                     'nexium', 'prilosec', 'protonix', 'prevacid', 'aciphex', 'dexilant','cytotec','misoprostol','omeprazol','pantoloc','previcid','gelusil','omeprezole'],
        'H2 Blockers': ['ranitidine', 'famotidine', 'cimetidine', 'zantac', 'pepcid', 'tagamet','rantidine','nizatidine','ranitadine'],
        'Other GI': ['Konsyl','Librax','Lomotil','Sennoside','Mesalamine','Domperidone','florastor','entocort','lactaid','colace','metoclopramide', 'ondansetron', 'loperamide', 'dicyclomine', 
                    'simethicone', 'bisacodyl', 'docusate', 'senna', 'fibercon', 'metamucil','Dulcolax', 'immodium', 'zelnorm','miralax','tums','Meclizine','Stool softener','Imodium','Citrucel','zofran','fiber','lactulose','pepto bismol','milk of magnesia','promethazine','senokot','guaifenesin','phenergan','robitussin','reglan','psyllium husk','bentyl','pancrelipase','hyoscyamine','amitiza','lubiprostone','carafate','sucralfate','fleet enema','ex-lax','glycolax','antivert','lonox','bonine','ducosate sodium','kaopectate','dramamine','levsin','tussin dm','metrocream','pamine','fenesin']
    },
    'Respiratory': {
        'Inhalers': ['Foradil','singular','singulair','qvar','albuterol', 'fluticasone', 'budesonide', 'formoterol', 
                    'symbicort', 'advair', 'tiotropium', 'spiriva','Flovent','pulmicort','combivent','atrovent','ventolin','proair','proventil','duoneb','salbutamol','xopenex','levalbuterol','aerobid','azmacort','beclomethasone spray'],
        'Other Respiratory': ['Serevent','Tessalon','asmanex','oxygen','montelukast', 'singulair', 'theophylline', 'ipratropium', 'zafirlukast','flunisolide','benzonatate','asthmanex','cingulair','metaproterenol']
    },
    'Urological': {
        'General Urological': ['tamsulosin', 'finasteride', 'oxybutynin', 'solifenacin', 
                                'tolterodine', 'flomax', 'proscar', 'detrol', 'dutasteride','terazosin','vesicare','avodart','viagra','cialis','uroxatral','alfuzosin','enablex','darifenacin','levitra','vardenafil','ditropan','oxybutin','sanctura','trospium','pyridium','phenazopyridine','toviaz','fesoterodine','uroxetrol','vosol','rapaflo','silodosin','desmopressin']
    },
    'Bone Health': {
        'Osteoporosis': ['Calcitonin','Synvisc','Miacalcin','alendronate', 'risedronate', 'ibandronate', 'zoledronic acid', 
                         'denosumab', 'teriparatide', 'raloxifene', 'fosamax','celebrex','actonel','Evista','boniva','Fosomax','forteo','reclast','fosamex','miacalcic','didrocal kit','didrocal','zometa','fortical']
    },
    'Neurological Disorders': {
        'AD & Dementia': ['donepezil', 'aricept', 'memantine', 'namenda', 
                                    'rivastigmine', 'exelon', 'galantamine', 'razadyne', 
                                    'ebixa', 'cognex','reminyl','razadine','ceriva','donezepil','excelon','nameda'],
        'Anticonvulsants': ['dilantin','gabapentin', 'pregabalin', 'topiramate', 'valproic acid', 
                            'carbamazepine', 'phenytoin', 'levetiracetam', 'lamotrigine','Neurontin','lyrica','lamictal','primidone','keppra','depakote','topamax','oxcarbazepine','tegretol','divalproex','trileptal'],
        'Parkinsons': ['carbidopa','levodopa', 'sinemet', 'pramipexole', 'ropinirole', 
                         'selegiline', 'amantadine','mirapex','requip','bromocriptine']
    },
    'Ophthalmologic': {
        'Glaucoma': ['Travitan','Brimonidine tartrate','Combigan','Brimonidine','Travoprost','xalatan', 'betoptics', 'cosopt', 'trusopt', 'travatan', 'alphagan','Timolol','dorzolamide','vigamox','timoptic','atropine','betoptic','betimol','pilocarpine','optipranolol','zylet','istalol','cyclopentolate'],
        'Eye Supplements': ['liquid tears','refresh','artificial tears','ocuvite', 'preservision','Restasis','Lumigan','systane','pred forte','nevanac','icaps','patanol','vitalux','i-caps','i caps','super vision supplement','eye vitamins','thera tears','pataday'],
        'Other Eye': ['Eye drops','Lotemax','Pred-forte','Acular','Omnipred','Polytrim','ofloxacin','fluorouracil','azopt','zymar','gatifloxacin','alrex','loteprednol','akwa']
    },
    'Allergy/Immunologic': {
        'Antihistamines': ['OTC Allergy Medication','zertec','allergy shots','Immunotherapy','Allergy relief','Loratidine','Phenylephrine','epinephrine','Chlorpheniramine maleate','Hydroxyzine','antihistamines','epipen','zyrtec', 'loratadine', 'allegra', 'benadryl', 'alovert','claritin','fexofenadine', "cetirizine",'diphenhydramine','clarinex','sudafed','atarax','benedryl','flu','claritan','xyzal','levocetirizine','pseudoephedrine','klor-tan'],
        'Nasal Sprays': ['Afrin','flonase', 'nasacort', 'astelin','Nasonex','debrox','astepro','azelastine','beconase','ponaris']
    },
    'Sleep/Sedation': {
        'Sleep Aids': ['CPAP','ambien', 'lunesta', 'melatonin','Zolpidem','zopiclone','provigil','doxylamine','restoril','gravol','dimenhydrinate','ambian tabs','zoplicone'],
        'Anti-Anxiety': ['clonazepam']
    },
    'Oncology': {
        'Chemotherapy': ['hydroxyurea']
    },
    'Smoking Cessation': {
        'Nicotine Receptor Modulators': ['chantix','varenicline']
    },
    'Neuromuscular': {
        'Myasthenia Gravis': ['pyridostigmine','mestinon']
    },
    'Weight Management': {
        'Appetite Suppressants': ['phentermine']
    },
    'Dermatological': {
        'Dermatitis': ['Elidel','Protopic','dovonex cream'],
        'Cancer': ['Efudex'],
        'Other Dermatological': ['liquid nitrogen']
    },
    'Hormone': {
        'Hormone Replacement': ['Prempro','Androderm','Testosterone','Premarin','Estradiol','estrace','androgel','vagifem','dhea','conjugated estrogen','medroxyprogesterone','tamoxifen','arimidex','casodex','bicalutamide','femara','letrozole','prometrium','progesterone','vivelle dot','testim','estring','megestrol','lupron','leuprolide','estropipate','progestin','estrogen','vivelle','femhrt','faslodex','fulvestrant','cenestin','estraderm','activella','targretin','provera','climara']
    },
}

# Function to classify medications using multi-tiered taxonomy
def classify_medication_v8(med_name):
    med_name_lower = med_name.lower() if isinstance(med_name, str) else ''
    taxonomy = {'General_Class': 'Other', 'Subclass': 'Other', 'Sub_Subclass': 'Other'}
    
    for general_class, subclasses in medication_taxonomy.items():
        if isinstance(subclasses, dict):
            for subclass, sub_items in subclasses.items():
                if isinstance(sub_items, dict):
                    for sub_subclass, meds in sub_items.items():
                        if any(((med if isinstance(med, str) else '')).strip().lower() in med_name_lower for med in meds):
                            taxonomy = {
                                'General_Class': general_class,
                                'Subclass': subclass,
                                'Sub_Subclass': sub_subclass
                            }
                            return taxonomy
                else:
                    if any(((med if isinstance(med, str) else '')).strip().lower() in med_name_lower for med in sub_items):
                        taxonomy = {
                            'General_Class': general_class,
                            'Subclass': subclass,
                            'Sub_Subclass': 'None'
                        }
                        return taxonomy
        else:
            # Handle cases where subclasses might not be dictionaries
            pass
    
    return taxonomy

# Apply the grouping function to the 'CMMED' column
subset_data = subset_data.dropna(subset=['CMMED'])
classification = subset_data['CMMED'].apply(classify_medication_v8)
subset_data[['General_Class', 'Subclass', 'Sub_Subclass']] = pd.DataFrame(classification.tolist(), index=subset_data.index)

# DROP UNCLASSIFIED MEDICATIONS (MED_Other) to prevent confounding
# Store stats before dropping
total_before_drop = len(subset_data)
other_count = len(subset_data[subset_data['Subclass'] == 'Other'])
subset_data = subset_data[subset_data['Subclass'] != 'Other'].copy()
print(f"\nDropped {other_count} unclassified medication records ({other_count/total_before_drop*100:.1f}%)")
print(f"Retained {len(subset_data)} classified medication records ({len(subset_data)/total_before_drop*100:.1f}%)")

# Create mappings for each taxonomy level (starting from 1 since we dropped 'Other')
general_class_mapping = {cls: i + 1 for i, cls in enumerate(sorted(subset_data['General_Class'].unique()))}

subclass_mapping = {sub: i + 1 for i, sub in enumerate(sorted(subset_data['Subclass'].unique()))}

sub_subclass_mapping = {sub_sub: i + 1 for i, sub_sub in enumerate(sorted(subset_data['Sub_Subclass'].unique()))}

# Apply the mappings to encode the taxonomy
subset_data['General_Class_Encoded'] = subset_data['General_Class'].map(general_class_mapping)
subset_data['Subclass_Encoded'] = subset_data['Subclass'].map(subclass_mapping)
subset_data['Sub_Subclass_Encoded'] = subset_data['Sub_Subclass'].map(sub_subclass_mapping)

# Save the final dataset with the encoded columns
output_path = 'processed/encoded_medicines.csv'
subset_data['SCANDATE'] = subset_data['VISDATE'].copy()

# Save files with error handling
try:
    subset_data[['RID', 'CMMED', 'General_Class_Encoded', 'Subclass_Encoded', 
                 'Sub_Subclass_Encoded', 'SCANDATE']].to_csv(output_path, index=False)
    print(f"Successfully saved encoded data to {output_path}")
except Exception as e:
    print(f"Error saving encoded data: {e}")

# Save the mappings to separate CSV files
mapping_path_general = 'processed/general_class_mapping.csv'
pd.DataFrame(list(general_class_mapping.items()), columns=['General_Class', 'Encoding']).to_csv(mapping_path_general, index=False)

mapping_path_subclass = 'processed/subclass_mapping.csv'
pd.DataFrame(list(subclass_mapping.items()), columns=['Subclass', 'Encoding']).to_csv(mapping_path_subclass, index=False)

mapping_path_sub_subclass = 'processed/sub_subclass_mapping.csv'
pd.DataFrame(list(sub_subclass_mapping.items()), columns=['Sub_Subclass', 'Encoding']).to_csv(mapping_path_sub_subclass, index=False)

# Output the paths for all encoded files
print(f"Encoded CSV Output: {output_path}")
print(f"General Class Mapping CSV Output: {mapping_path_general}")
print(f"Subclass Mapping CSV Output: {mapping_path_subclass}")
print(f"Sub-Subclass Mapping CSV Output: {mapping_path_sub_subclass}")

# Final classification statistics
print("\nFinal Classification Summary:")
print(f"Total medication records saved: {len(subset_data)}")
print(f"Unique medication subclasses: {len(subclass_mapping)}")
print(f"Unique medication general classes: {len(general_class_mapping)}")
print("\nNote: All unclassified medications (MED_Other) have been excluded to prevent confounding.")

# Build a supplement-ready hierarchy table of medication taxonomy (without Sub_Subclass)
aggregation = {}
for general_class, subclasses in medication_taxonomy.items():
    if isinstance(subclasses, dict):
        for subclass, sub_items in subclasses.items():
            key = (general_class, subclass)
            if key not in aggregation:
                aggregation[key] = set()
            if isinstance(sub_items, dict):
                for _, meds in sub_items.items():
                    for m in meds:
                        if isinstance(m, str):
                            aggregation[key].add(m)
            else:
                for m in sub_items:
                    if isinstance(m, str):
                        aggregation[key].add(m)

taxonomy_records = []
for (general_class, subclass), meds_set in aggregation.items():
    meds_list = sorted(set(meds_set), key=lambda s: s.lower())
    taxonomy_records.append({
        'General_Class': general_class,
        'Subclass': subclass,
        'Medications': '; '.join(meds_list)
    })

taxonomy_table = pd.DataFrame(taxonomy_records).sort_values(['General_Class', 'Subclass'])
taxonomy_csv_path = 'processed/medication_taxonomy_table.csv'
taxonomy_html_path = 'processed/medication_taxonomy_table.html'

try:
    taxonomy_table.to_csv(taxonomy_csv_path, index=False)
    taxonomy_table.to_html(taxonomy_html_path, index=False)
    print(f"\nPublication supplement table saved to: {taxonomy_csv_path} and {taxonomy_html_path}")
except Exception as e:
    print(f"Error saving taxonomy supplement table: {e}")