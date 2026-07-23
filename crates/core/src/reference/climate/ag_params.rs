use phf::phf_map;

// Per-index impact estimates
pub const CDD_YEARLY_WHEAT_ESTIMATE: f32 = 0.006107;
pub const CDD_YEARLY_RICE_ESTIMATE: f32 = 0.025827;
pub const CDD_YEARLY_MAIZE_ESTIMATE: f32 = 0.01979;

pub const CXDD_YEARLY_WHEAT_ESTIMATE: f32 = 0.006571;
pub const CXDD_YEARLY_RICE_ESTIMATE: f32 = 0.026962;
pub const CXDD_YEARLY_MAIZE_ESTIMATE: f32 = 0.020766;

// Per-index impact estimates
pub const CDD_INDEX_WHEAT_ESTIMATE: f32 = -0.000004;
pub const CDD_INDEX_RICE_ESTIMATE: f32 = -0.000025;
pub const CDD_INDEX_MAIZE_ESTIMATE: f32 = -0.000004;

pub struct NationalCropWeights {
    pub maize: f32,
    pub rice: f32,
    pub wheat: f32,
}

const DEFAULT_CROP_WEIGHTS: NationalCropWeights = NationalCropWeights {
    maize: 0.333,
    rice: 0.333,
    wheat: 0.333,
};

pub const CDD_YEARLY_CROP_ESTIMATES: NationalCropWeights = NationalCropWeights {
    maize: CDD_YEARLY_MAIZE_ESTIMATE,
    rice: CDD_YEARLY_RICE_ESTIMATE,
    wheat: CDD_YEARLY_WHEAT_ESTIMATE,
};

pub const CXDD_YEARLY_CROP_ESTIMATES: NationalCropWeights = NationalCropWeights {
    maize: CXDD_YEARLY_MAIZE_ESTIMATE,
    rice: CXDD_YEARLY_RICE_ESTIMATE,
    wheat: CXDD_YEARLY_WHEAT_ESTIMATE,
};

pub const CDD_INDEX_CROP_ESTIMATES: NationalCropWeights = NationalCropWeights {
    maize: CDD_INDEX_MAIZE_ESTIMATE,
    rice: CDD_INDEX_RICE_ESTIMATE,
    wheat: CDD_INDEX_WHEAT_ESTIMATE,
};

pub const CXDD_INDEX_CROP_ESTIMATES: NationalCropWeights = NationalCropWeights {
    maize: 0.0,
    rice: 0.0,
    wheat: 0.0,
};

const DEFAULT_CDD_CROP_VALUES: NationalCropWeights = NationalCropWeights {
    maize: 0.0,
    rice: 0.0,
    wheat: 0.0,
};

pub const CDD_CROP_VALUES: phf::Map<&'static str, NationalCropWeights> = phf_map! {
    "ARG" => NationalCropWeights {
      maize: -38.0489005961184,
      rice: -49.8820147171806,
      wheat: -11.1389024149692,
    },
    "AUS" => NationalCropWeights {
      maize: -38.0604482706578,
      rice: -49.6929032231287,
      wheat: -11.348452853592,
    },
    "BRA"=> NationalCropWeights {
      maize: -38.531404386616,
      rice: -50.4581519921146,
      wheat: -11.4476109570311,
    },
    "BGR"=> NationalCropWeights {
      maize: -38.0886980694688,
      rice: -49.4123634046303,
      wheat: -10.8957325721854,
    },
    "HUN"=> NationalCropWeights {
      maize: -37.8923373905435,
      wheat: -10.7731377806262,
      rice: 0.0,
    },
    "IND"=> NationalCropWeights {
      maize: -38.6683135016376,
      rice: -50.4657266995613,
      wheat: -10.9685518351655,
    },
    "ISR"=> NationalCropWeights {
      maize: -38.6465601756504,
      wheat: -11.0997014126484,
      rice: 0.0,
    },
    "ROU"=> NationalCropWeights {
      maize: -38.2500159951297,
      wheat: -10.9852694864019,
      rice: 0.0,
    },
    "RUS"=> NationalCropWeights {
      maize: -38.2037609740528,
      rice: -49.8640817163549,
      wheat: -11.3233237595455,
    },
    "THA"=> NationalCropWeights {
      maize: -38.3782406812006,
      rice: -50.5446543798874,
      wheat: -10.021507179372,
    },
    "UKR"=> NationalCropWeights {
      maize: -38.3613363432157,
      wheat: -11.0651780007127,
      rice: 0.0,
    },
    "USA"=> NationalCropWeights {
      maize: -37.5869376157853,
      rice: -49.5887020605306,
      wheat: -10.7425222595814,
    },
    "year"=> NationalCropWeights {
      maize: 0.0197902721416242,
      rice: 0.025826967540696,
      wheat: 0.0061070027493708,
    },
    "cdd"=> NationalCropWeights {
        maize: -4.46411887464599e-6,
      rice: -2.49939444391363e-5,
      wheat: -0.0001163897867293,
    },
    "LTU"=> NationalCropWeights {
      wheat: -11.0031310916395,
      maize: 0.0,
      rice: 0.0,
    },
    "POL"=> NationalCropWeights {
      wheat: -10.8802736174415,
      maize: 0.0,
      rice: 0.0,
    },
};

pub fn get_cdd_crop_values(country: &str) -> &NationalCropWeights {
    CDD_CROP_VALUES
        .get(country)
        .unwrap_or(&DEFAULT_CDD_CROP_VALUES)
}

const DEFAULT_CXDD_CROP_VALUES: NationalCropWeights = NationalCropWeights {
    maize: 0.0,
    rice: 0.0,
    wheat: 0.0,
};

pub const CXDD_CROP_VALUES: phf::Map<&'static str, NationalCropWeights> = phf_map! {
    "ARG"=> NationalCropWeights {
      maize: -39.8608384791876,
      rice: -52.1175848143291,
      wheat: -11.9934079647692
    },
    "AUS"=> NationalCropWeights {
      maize: -39.7357373133152 ,
      rice: -51.86732711052 ,
      wheat: -12.0955256701341,
    },
    "BRA"=> NationalCropWeights {
      maize: -40.218294121002 ,
      rice: -52.6092028953173 ,
      wheat: -12.2949631679722,
    },
    "BGR"=> NationalCropWeights {
      maize: -39.9162728533279 ,
      rice: -51.6051268752724 ,
      wheat: -11.6931906298432
    },
    "HUN"=> NationalCropWeights {maize: -39.7304161043605 , rice: 0.0, wheat: -11.5819566574985 },
    "IND"=> NationalCropWeights {
      maize: -40.3933829808387 ,
      rice: -52.6202718934997 ,
      wheat: -11.6768262323116
    },
    "ISR"=> NationalCropWeights {maize: -39.9899107130726 , rice: 0.0, wheat: -11.1172428702865 },
    "ROU"=> NationalCropWeights {maize: -40.0910658511114 , rice: 0.0, wheat: -11.7942079448484 },
    "RUS"=> NationalCropWeights {
      maize: -40.0038808529967 ,
      rice: -52.0366491040825 ,
      wheat: -12.079416546518
    },
    "THA"=> NationalCropWeights {
      maize: -40.098824717163 ,
      rice: -52.7389104286115 ,
      wheat: -10.8110471042261
    },
    "UKR"=> NationalCropWeights {
      maize: -40.2007989000116,
      wheat: -11.8695378368934,
      rice: 0.0,
    },
    "USA"=> NationalCropWeights {
      maize: -39.4159263600224,
      rice: -51.7706163860741,
      wheat: -11.5684236748059,
    },
    "LTU"=> NationalCropWeights {
      wheat: -11.8029901080071,
      maize: 0.0,
      rice: 0.0,
    },
    "POL"=> NationalCropWeights {
      wheat: -11.6914711979464,
      maize: 0.0,
      rice: 0.0,
    },
};

pub fn get_cxdd_crop_values(country: &str) -> &NationalCropWeights {
    CXDD_CROP_VALUES
        .get(country)
        .unwrap_or(&DEFAULT_CXDD_CROP_VALUES)
}

pub const NATIONAL_CROP_WEIGHTS: phf::Map<&'static str, NationalCropWeights> = phf_map! {
    "ALB" => NationalCropWeights {
      maize: 0.6410390332040676,
      wheat: 0.3589609667959324,
      rice: 0.0
    },
    "DZA" => NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "ATG" => NationalCropWeights { maize: 1.0, wheat: 0.0, rice: 0.0 },
    "ARM" => NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "AUS"=> NationalCropWeights {
      maize: 0.01117146591830722,
      rice: 0.01095655757873038,
      wheat: 0.9778719765029624
    },
    "AUT" => NationalCropWeights {
      maize: 0.5646767958856656,
      wheat: 0.4353232041143345,
      rice: 0.0
    },
    "AZE" => NationalCropWeights {
      maize: 0.1431495750864252,
      rice: 0.04089671821252316,
      wheat: 0.8159537067010516
    },
    "BHR" => NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "BGD" => NationalCropWeights {
      maize: 0.05755104560213924,
      rice: 0.9211411643729708,
      wheat: 0.021307790024889965
    },
    "BRB" => NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "BLR" => NationalCropWeights {
      maize: 0.41858383877325067,
      wheat: 0.5814161612267493,
      rice: 0.0
    },
    "BEL" => NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "BLZ" => NationalCropWeights {
      maize: 0.8468822901427385,
      rice: 0.15311770985726147,
      wheat: 0.0
    },
    "BEN" => NationalCropWeights {
      maize: 0.6241778681827075,
      rice: 0.3758221318172925,
      wheat: 0.0
    },
    "BTN"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 },
    "BOL"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "BIH"=> NationalCropWeights {
      maize: 0.7856408670544249,
      wheat: 0.21435913294557513,
      rice: 0.0
    },
    "BRA"=> NationalCropWeights {
      maize: 0.8012204681589821,
      rice: 0.11885037319651567,
      wheat: 0.07992915864450227
    },
    "BRN"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 },
    "BGR"=> NationalCropWeights {
      maize: 0.3059514921983482,
      rice: 0.013174654142732207,
      wheat: 0.6808738536589196
    },
    "BFA"=> NationalCropWeights { maize: 1.0, wheat: 0.0, rice: 0.0 },
    "BDI"=> NationalCropWeights {
      maize: 0.5703721122191402,
      rice: 0.4296278877808598,
      wheat: 0.0
    },
    "CPV"=> NationalCropWeights { maize: 1.0, wheat: 0.0, rice: 0.0 },
    "CAN"=> NationalCropWeights {
      maize: 0.26799760425923763,
      wheat: 0.7320023957407624,
      rice: 0.0
    },
    "TCD"=> NationalCropWeights {
      maize: 0.3944642066959664,
      rice: 0.602295428475686,
      wheat: 0.0032403648283476608
    },
    "CHL"=> NationalCropWeights {
      maize: 0.29822341588778656,
      rice: 0.07135432881170771,
      wheat: 0.6304222553005058
    },
    "CHN"=> NationalCropWeights {
      maize: 0.21906744069547315,
      rice: 0.17262636167183648,
      wheat: 0.10830619763269037
    },
    "COL"=> NationalCropWeights {
      maize: 0.32186583741800195,
      rice: 0.6756488947006256,
      wheat: 0.002485267881372393
    },
    "CRI"=> NationalCropWeights {
      maize: 0.05063712093773666,
      rice: 0.9493628790622634,
      wheat: 0.0
    },
    "HRV"=> NationalCropWeights {
      maize: 0.6834935737272382,
      wheat: 0.3165064262727618,
      rice: 0.0
    },
    "CYP"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "CZE"=> NationalCropWeights {
      maize: 0.11784364260404091,
      wheat: 0.8821563573959591,
      rice: 0.0
    },
    "CIV"=> NationalCropWeights {
      maize: 0.48247549791867556,
      rice: 0.5175245020813244,
      wheat: 0.0
    },
    "DNK"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "ECU"=> NationalCropWeights {
      maize: 0.569788358960722,
      rice: 0.42595956473720076,
      wheat: 0.0042520763020771945
    },
    "EGY"=> NationalCropWeights {
      maize: 0.38619633608291076,
      wheat: 0.6138036639170893,
      rice: 0.0
    },
    "SLV"=> NationalCropWeights {
      maize: 0.9745116509000106,
      rice: 0.025488349099989465,
      wheat: 0.0
    },
    "EST"=> NationalCropWeights{ wheat: 1.0, maize: 0.0, rice: 0.0 },
    "FJI"=> NationalCropWeights {
      maize: 0.2786139855754155,
      rice: 0.7213860144245845,
      wheat: 0.0
    },
    "FIN"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "FRA"=> NationalCropWeights{
      maize: 0.26592763091341554,
      rice: 0.001054310137305205,
      wheat: 0.7330180589492793
    },
    "GEO"=> NationalCropWeights {
      maize: 0.636707788865459,
      wheat: 0.36329221113454097,
      rice: 0.0
    },
    "DEU"=> NationalCropWeights{
      maize: 0.15467555450034237,
      wheat: 0.8453244454996577,
      rice: 0.0
    },
    "GHA"=> NationalCropWeights {
      maize: 0.7086905712278037,
      rice: 0.2913094287721964,
      wheat: 0.0
    },
    "GRC"=> NationalCropWeights {
      maize: 0.47313417389409995,
      rice: 0.0839765288189225,
      wheat: 0.44288929728697757
    },
    "GRD"=> NationalCropWeights{ maize: 1.0, wheat: 0.0, rice: 0.0 },
    "GIN"=> NationalCropWeights {
      maize: 0.12605647496131975,
      rice: 0.8739435250386802,
      wheat: 0.0
    },
    "GUY"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "HUN"=> NationalCropWeights {
      maize: 0.5137362060594928,
      wheat: 0.4862637939405073,
      rice: 0.0
    },
    "ISL"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "IDN"=> NationalCropWeights {
      maize: 0.2559594983722778,
      rice: 0.7440405016277222,
      wheat: 0.0
    },
    "IRN"=> NationalCropWeights {
      maize: 0.017488443736797803,
      rice: 0.44248640892490454,
      wheat: 0.5400251473382977
    },
    "IRQ"=> NationalCropWeights{
      maize: 0.07185384344427997,
      rice: 0.10493954251908207,
      wheat: 0.823206614036638
    },
    "IRL"=> NationalCropWeights{ wheat: 1.0, maize: 0.0, rice: 0.0 },
    "ISR"=> NationalCropWeights {
      maize: 0.5549025959002359,
      wheat: 0.445097404099764,
      rice: 0.0
    },
    "ITA"=> NationalCropWeights {
      maize: 0.33414172635818534,
      rice: 0.15632964798978916,
      wheat: 0.5095286256520255
    },
    "JAM"=> NationalCropWeights { maize: 1.0, wheat: 0.0, rice: 0.0 },
    "JPN"=> NationalCropWeights {
      rice: 0.979948314478748,
      wheat: 0.020051685521251975,
      maize: 0.0
    },
    "JOR"=> NationalCropWeights{ wheat: 1.0, maize: 0.0, rice: 0.0 },
    "KAZ"=> NationalCropWeights {
      maize: 0.056560023761039725,
      rice: 0.03504727045714017,
      wheat: 0.9083927057818201
    },
    "KEN"=> NationalCropWeights {
      maize: 0.8453227849707664,
      rice: 0.061027181829981694,
      wheat: 0.09365003319925193
    },
    "KWT"=> NationalCropWeights {
      maize: 0.9976344205721919,
      wheat: 0.002365579427808082,
      rice: 0.0
    },
    "KGZ"=> NationalCropWeights {
      maize: 0.4907968447631229,
      rice: 0.14118896330053962,
      wheat: 0.36801419193633744
    },
    "LVA"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "LBN"=> NationalCropWeights {
      maize: 0.014104624174254598,
      wheat: 0.9858953758257454,
      rice: 0.0
    },
    "LSO"=> NationalCropWeights {
      maize: 0.8721818958155423,
      wheat: 0.12781810418445771,
      rice: 0.0
    },
    "LTU"=> NationalCropWeights {
      maize: 0.022221236806111534,
      wheat: 0.9777787631938885,
      rice: 0.0
    },
    "LUX"=> NationalCropWeights {
      maize: 0.008100654946570147,
      wheat: 0.9918993450534298,
      rice: 0.0
    },
    "MDG"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 },
    "MYS"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 },
    "MDV"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "MLI"=> NationalCropWeights {
      maize: 0.5075756052104075,
      rice: 0.4924243947895926,
      wheat: 0.0
    },
    "MLT"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "MUS"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "MEX"=> NationalCropWeights {
      maize: 0.8671163748266771,
      rice: 0.008306152869182213,
      wheat: 0.12457747230414068
    },
    "MNG"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "MAR"=> NationalCropWeights {
      maize: 0.01368307794942417,
      rice: 0.007888243754374639,
      wheat: 0.9784286782962012
    },
    "MOZ"=> NationalCropWeights {
      maize: 0.779218076787035,
      rice: 0.20664948652544574,
      wheat: 0.014132436687519172
    },
    "NAM"=> NationalCropWeights {
      maize: 0.8206706386331183,
      wheat: 0.17932936136688168,
      rice: 0.0
    },
    "NPL"=> NationalCropWeights {
      maize: 0.265795808549019,
      rice: 0.5183311593961838,
      wheat: 0.21587303205479716
    },
    "NCL"=> NationalCropWeights {
      maize: 0.9917852597477728,
      wheat: 0.008214740252227236,
      rice: 0.0
    },
    "NZL"=> NationalCropWeights {
      maize: 0.31325870889850677,
      wheat: 0.6867412911014932,
      rice: 0.0
    },
    "NIC"=> NationalCropWeights {
      maize: 0.5248656979722605,
      rice: 0.47513430202773954,
      wheat: 0.0
    },
    "NGA"=> NationalCropWeights {
      maize: 0.4344618863823178,
      rice: 0.5655381136176822,
      wheat: 0.0
    },
    "NOR"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "OMN"=> NationalCropWeights {
      maize: 0.5209008619654734,
      wheat: 0.47909913803452664,
      rice: 0.0
    },
    "PAK"=> NationalCropWeights {
      maize: 0.1454505291726674,
      rice: 0.350852043497768,
      wheat: 0.5036974273295646
    },
    "PAN"=> NationalCropWeights{
      maize: 0.23217111969761192,
      rice: 0.767828880302388,
      wheat: 0.0
    },
    "PRY"=> NationalCropWeights{
      maize: 0.7555787485858139,
      rice: 0.06742638926726784,
      wheat: 0.17699486214691834
    },
    "PER"=> NationalCropWeights {
      maize: 0.44391132055067195,
      rice: 0.5061353044369586,
      wheat: 0.049953375012369475
    },
    "POL"=> NationalCropWeights {
      maize: 0.3164484063119118,
      wheat: 0.6835515936880882,
      rice: 0.0
    },
    "PRT"=> NationalCropWeights {
      maize: 0.6680407644255346,
      rice: 0.26434864206390907,
      wheat: 0.06761059351055634
    },
    "QAT"=> NationalCropWeights {
      maize: 0.9636623748211731,
      wheat: 0.036337625178826896,
      rice: 0.0
    },
    "ROU"=> NationalCropWeights {
      maize: 0.5550764819210804,
      wheat: 0.44492351807891967,
      rice: 0.0
    },
    "RWA"=> NationalCropWeights {
      maize: 0.6469034529318322,
      rice: 0.32062243341319346,
      wheat: 0.03247411365497443
    },
    "KNA"=> NationalCropWeights{ wheat: 0.0, maize: 0.0, rice: 0.0 },
    "LCA"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "VCT"=> NationalCropWeights { maize: 1.0, wheat: 0.0, rice: 0.0 },
    "WSM"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "SAU"=> NationalCropWeights {
      maize: 0.0504925781457388,
      wheat: 0.9495074218542612,
      rice: 0.0
    },
    "SEN"=> NationalCropWeights {
      maize: 0.4390266956752785,
      rice: 0.5609733043247215,
      wheat: 0.0
    },
    "SRB"=> NationalCropWeights {
      maize: 0.6444426046712349,
      wheat: 0.3555573953287651,
      rice: 0.0
    },
    "SGP"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "SVK"=> NationalCropWeights {
      maize: 0.3472018762284208,
      wheat: 0.6527981237715792,
      rice: 0.0
    },
    "SVN"=> NationalCropWeights {
      maize: 0.6827780663636834,
      wheat: 0.31722193363631657,
      rice: 0.0
    },
    "ZAF"=> NationalCropWeights {
      maize: 0.8061347850805988,
      wheat: 0.19386521491940126,
      rice: 0.0
    },
    "ESP"=> NationalCropWeights {
      maize: 0.33055778136338526,
      rice: 0.0745833427695145,
      wheat: 0.5948588758671002
    },
    "LKA"=> NationalCropWeights {
      maize: 0.08124969128812601,
      rice: 0.918750308711874,
      wheat: 0.0
    },
    "SUR"=> NationalCropWeights{
      maize: 0.0012438862387536985,
      rice: 0.9987561137612463,
      wheat: 0.0
    },
    "SWE"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "CHE"=> NationalCropWeights {
      maize: 0.16401904291058875,
      wheat: 0.8359809570894112,
      rice: 0.0
    },
    "TJK"=> NationalCropWeights {
      maize: 0.18746863903965558,
      rice: 0.1788810026314479,
      wheat: 0.6336503583288965
    },
    "THA"=> NationalCropWeights {
      maize: 0.11288804500922574,
      rice: 0.8871119549907742,
      wheat: 0.0
    },
    "TLS"=> NationalCropWeights {
      maize: 0.4357149046324477,
      rice: 0.5642850953675523,
      wheat: 0.0
    },
    "TGO"=> NationalCropWeights {
      maize: 0.8503256203362403,
      rice: 0.14967437966375977,
      wheat: 0.0
    },
    "TON"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "TTO"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 },
    "TUN"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "UKR"=> NationalCropWeights {
      maize: 0.5596688072890512,
      rice: 0.0011372651168648556,
      wheat: 0.43919392759408393
    },
    "URY"=> NationalCropWeights {
      maize: 0.2653843739132091,
      rice: 0.39706234026598647,
      wheat: 0.33755328582080446
    },
    "UZB"=> NationalCropWeights {
      maize: 0.11405816032700264,
      rice: 0.15562270343429385,
      wheat: 0.7303191362387035
    },
    "VNM"=> NationalCropWeights {
      maize: 0.09074803987247441,
      rice: 0.9092519601275256,
      wheat: 0.0
    },
    "YEM"=> NationalCropWeights {
      maize: 0.35104723218667827,
      wheat: 0.6489527678133218,
      rice: 0.0
    },
    "ZMB"=> NationalCropWeights {
      maize: 0.9022761745559053,
      rice: 0.09772382544409475,
      wheat: 0.0
    },
    "ZWE"=> NationalCropWeights {
      maize: 0.8133942212315024,
      rice: 0.0026625321734950486,
      wheat: 0.18394324659500255
    },
    "COK"=> NationalCropWeights { wheat: 0.0, maize: 0.0, rice: 0.0 },
    "DOM"=> NationalCropWeights {
      maize: 0.057099867539562275,
      rice: 0.9429001324604377,
      wheat: 0.0
    },
    "NLD"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "NER"=> NationalCropWeights {
      maize: 0.02797418941246952,
      rice: 0.9522023330267912,
      wheat: 0.01982347756073921
    },
    "MKD"=> NationalCropWeights {
      maize: 0.32535986874007744,
      rice: 0.08437125655690698,
      wheat: 0.5902688747030156
    },
    "PSE"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "PHL"=> NationalCropWeights {
      maize: 0.2713789211152045,
      rice: 0.7286210788847954,
      wheat: 0.0
    },
    "MDA"=> NationalCropWeights{
      maize: 0.5745736839771936,
      wheat: 0.4254263160228064,
      rice: 0.0
    },
    "RUS"=> NationalCropWeights {
      maize: 0.14176969643671036,
      rice: 0.01768445774904136,
      wheat: 0.8405458458142483
    },
    "TUR"=> NationalCropWeights {
      maize: 0.21641660834656534,
      rice: 0.07779708193952806,
      wheat: 0.7057863097139065
    },
    "GBR"=> NationalCropWeights { wheat: 1.0, maize: 0.0, rice: 0.0 },
    "USA"=> NationalCropWeights {
      maize: 0.8305073047275171,
      rice: 0.03595444974614065,
      wheat: 0.13353824552634222
    },
    "GMB"=> NationalCropWeights {
      maize: 0.3437532468432026,
      rice: 0.6562467531567974,
      wheat: 0.0
    },
    "KOR"=> NationalCropWeights { rice: 1.0, wheat: 0.0, maize: 0.0 }
};

pub fn get_national_crop_weights(country: &str) -> &NationalCropWeights {
    NATIONAL_CROP_WEIGHTS
        .get(country)
        .unwrap_or(&DEFAULT_CROP_WEIGHTS)
}
