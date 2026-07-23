use phf::phf_map;

pub struct FloodParams {
    pub residential_damage: [f32; 9],
    pub commercial_damage: [f32; 9],
    pub industrial_damage: [f32; 9],
    pub transport_damage: [f32; 9],
    pub infrastructure_damage: [f32; 9],
    pub agriculture_damage: [f32; 9],
}

pub const FLOOD_DAMAGE_THRESHOLDS: [f32; 9] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0];

pub const FLOOD_PARAMS: phf::Map<&'static str, FloodParams> = phf_map! {
    "Europe" => FloodParams {
        residential_damage: [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 0.85, 0.95, 1.0],
        commercial_damage: [0.0, 0.15, 0.3, 0.45, 0.55, 0.75, 0.9, 1.0, 1.0],
        industrial_damage: [0.0, 0.15, 0.27, 0.4, 0.52, 0.7, 0.85, 1.0, 1.0],
        transport_damage: [0.0, 0.316, 0.542, 0.7016, 0.8316, 1.0, 1.0, 1.0, 1.0],
        infrastructure_damage: [0.0, 0.25, 0.42, 0.55, 0.65, 0.8, 0.9, 1.0, 1.0],
        agriculture_damage: [0.0, 0.3, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.0],
    },
    "North America" => FloodParams {
        residential_damage: [0.20180454348279775, 0.44326985656702733, 0.582754693231323, 0.6825219115800791, 0.783957148211158, 0.8543489222586077, 0.9236701008849423, 0.9585227725931244, 1.0],
        commercial_damage: [0.018404907975460124, 0.23926380368098157, 0.37423312883435583, 0.4662576687116564, 0.5521472392638037, 0.6871165644171779, 0.8220858895705521, 0.9079754601226995, 1.0],
        industrial_damage: [0.025714285714285714, 0.3228571428571429, 0.5114285714285715, 0.6371428571428571, 0.7400000000000001, 0.8599999999999999, 0.937142857142857, 0.9800000000000001, 1.0],
        transport_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        infrastructure_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        agriculture_damage: [0.018575387611414872, 0.26779766808978106, 0.4736773774748458, 0.5505607847379999, 0.6021613029402708, 0.7600570910210638, 0.8740946024295585, 0.954075572089204, 1.0],
    },
    "South America" => FloodParams {
        residential_damage: [0.0, 0.4908859507565592, 0.7112940666211631, 0.842026010706856, 0.9493690958164642, 0.9836369770580298, 1.0, 1.0, 1.0],
        commercial_damage: [0.0, 0.6114775866089124, 0.8395310944604374, 0.9235884574511017, 0.9919724770642202, 1.0, 1.0, 1.0, 1.0],
        industrial_damage: [0.0, 0.6670194003527335, 0.8887125220458554, 0.9467372134038801, 1.0, 1.0, 1.0, 1.0, 1.0],
        transport_damage: [0.0, 0.08771929824561404, 0.1754385964912281, 0.5964912280701755, 0.8421052631578947, 1.0, 1.0, 1.0, 1.0],
        infrastructure_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        agriculture_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "Asia" => FloodParams {
        residential_damage: [0.0, 0.32655650202936004, 0.49405032354045164, 0.6165721244538469, 0.7207117644253294, 0.8695282125617129, 0.93148708390482, 0.9836041480130159, 1.0],
        commercial_damage: [0.0, 0.37678962318356757, 0.5376816186459964, 0.6593366839515573, 0.7628452324986368, 0.8833486558294331, 0.9418548948983457, 0.9807593802222822, 1.0],
        industrial_damage: [0.0, 0.2831815239036498, 0.4816156531424614, 0.6292188938464128, 0.7172405880206049, 0.8566750297600425, 0.9085770039670126, 0.9553274630224672, 1.0],
        transport_damage: [0.0, 0.35751633986928105, 0.5718954248366014, 0.7333333333333334, 0.8472222222222223, 1.0, 1.0, 1.0, 1.0],
        infrastructure_damage: [0.0, 0.2144369063772049, 0.37275440976933516, 0.6039348710990502, 0.709659090909091, 0.808409090909091, 0.887159090909091, 0.96875, 1.0],
        agriculture_damage: [0.0, 0.135, 0.37, 0.524, 0.558, 0.66, 0.834, 0.9879999999999999, 1.0],
    },
    "Africa" => FloodParams {
        residential_damage: [0.0, 0.2199254009329045, 0.37822684609280993, 0.5305890817622265, 0.635636732936187, 0.8169397798520446, 0.9034346883994495, 0.9571521730123568, 1.0],
        commercial_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        industrial_damage: [0.0, 0.06268204283360791, 0.2471960461285008, 0.4033299835255354, 0.49448863261943987, 0.684652388797364, 0.9185897858319604, 1.0, 1.0],
        transport_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        infrastructure_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        agriculture_damage: [0.0, 0.24287356321839082, 0.47183908045977013, 0.7413793103448275, 0.9166666666666666, 1.0, 1.0, 1.0, 1.0],
    },
    "Oceania" => FloodParams {
        residential_damage: [0.0, 0.47541811916699134, 0.6403931243607716, 0.7146146615077392, 0.7877263475521771, 0.9287798842694879, 0.9673818525004227, 0.9827954437588989, 1.0],
        commercial_damage: [0.0, 0.23895357454908572, 0.4811996817075454, 0.6737950914310677, 0.8645833333333333, 1.0, 1.0, 1.0, 1.0],
        industrial_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        transport_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        infrastructure_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        agriculture_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "Global" => FloodParams {
        residential_damage: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        commercial_damage: [0.0, 0.3232969176045094, 0.506529104729667, 0.6345955803090766, 0.7443096564319989, 0.8640930440493222, 0.9327881568937796, 0.9777469680689963, 1.0],
        industrial_damage: [0.0, 0.2971480219894268, 0.4797905585490779, 0.6032857895837371, 0.694345844128009, 0.8202654837114812, 0.9228619293883661, 0.9870654926044935, 1.0],
        transport_damage: [0.0, 0.25396743492718726, 0.4296668959981654, 0.6771637426900585, 0.8403313840155945, 1.0, 1.0, 1.0, 1.0],
        infrastructure_damage: [0.0, 0.23221845318860246, 0.39637720488466754, 0.5769674355495251, 0.6798295454545455, 0.8042045454545454, 0.8935795454545454, 0.984375, 1.0],
        agriculture_damage: [0.0, 0.23641780782704297, 0.466379114483654, 0.6164850237707068, 0.7067069924017344, 0.817514272755266, 0.9145236506073896, 0.985518893022301, 1.0],
    },
};

pub fn continent_to_flood_params(continent: &str) -> &FloodParams {
    match continent {
        "Europe" => &FLOOD_PARAMS["Europe"],
        "North America" => &FLOOD_PARAMS["North America"],
        "South America" => &FLOOD_PARAMS["South America"],
        "Asia" => &FLOOD_PARAMS["Asia"],
        "Africa" => &FLOOD_PARAMS["Africa"],
        "Oceania" => &FLOOD_PARAMS["Oceania"],
        "Global" => &FLOOD_PARAMS["Global"],
        _ => &FLOOD_PARAMS["Global"],
    }
}

// Returns the index of the flood level in the flood params array
pub fn _flood_level_to_index(flood_level: f32) -> usize {
    if flood_level > 6.0 {
        8
    } else if flood_level > 5.0 {
        7
    } else if flood_level > 4.0 {
        6
    } else if flood_level > 3.0 {
        5
    } else if flood_level > 2.0 {
        4
    } else if flood_level > 1.5 {
        3
    } else if flood_level > 1.0 {
        2
    } else if flood_level > 0.5 {
        1
    } else {
        0
    }
}

pub fn building_type_to_damage(building_type: &str, flood_params: &FloodParams) -> [f32; 9] {
    match building_type {
        "Residential buildings" => flood_params.residential_damage,
        "Commercial buildings" => flood_params.commercial_damage,
        "Industrial buildings" => flood_params.industrial_damage,
        "Transport" => flood_params.transport_damage,
        "Infrastructure - roads" => flood_params.infrastructure_damage,
        "Agriculture" => flood_params.agriculture_damage,
        _ => [0.0; 9],
    }
}
