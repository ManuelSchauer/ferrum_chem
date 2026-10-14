//! JPlogP (CDK 2.13 `JPlogPDescriptor`, default parameters).
//!
//! CDK makes every hydrogen explicit, re-perceives aromaticity with
//! `Aromaticity.cdkLegacy()` and sums one coefficient per atom, looked up by
//! an integer atom type code (charge, element, heavy neighbour count and a
//! two-digit environment class). An atom whose code has no coefficient makes
//! the whole value NaN (CDK prints `<code> not found`).
//!
//! Key symbols: `fn jplogp`, `fn type_code`, `JPLOGP_WEIGHTS`.

use crate::desc::cdk_logp::LogpView;
use crate::desc::cdk_perception::CdkView;
use crate::mol::molecule::Molecule;

/// Atom environment as seen by `JPlogPCalculator` (hydrogens explicit,
/// `cdkLegacy` aromaticity, Kekulé bond orders).
struct Jp<'a> {
    v: &'a LogpView,
    arom: &'a [bool],
    arom_bond: &'a [bool],
}

/// A neighbour: atom (`None` for an implicit hydrogen), bond order, aromatic bond.
type Nb = (Option<usize>, u8, bool);

/// CDK `JPlogPDescriptor` value (NaN when an atom type has no coefficient).
pub(crate) fn jplogp(
    mol: &Molecule,
    v: &LogpView,
    ctx: &mut crate::desc::context::CalculationContext,
) -> f64 {
    CdkView::with_legacy_cached(mol, ctx, |view, _| {
        let jp = Jp {
            v,
            arom: &view.arom_atom,
            arom_bond: &view.arom_bond,
        };
        jp.calc()
    })
}

#[inline]
fn is_polar(z: u8) -> bool {
    matches!(z, 7 | 8 | 15 | 16)
}

#[inline]
fn electron_withdrawing(z: u8) -> bool {
    matches!(z, 7 | 8 | 9 | 16 | 17 | 35 | 53)
}

fn coefficient(code: i32) -> Option<f64> {
    JPLOGP_WEIGHTS
        .binary_search_by_key(&code, |&(k, _)| k)
        .ok()
        .map(|i| JPLOGP_WEIGHTS[i].1)
}

impl Jp<'_> {
    /// Neighbours of a real atom, heavy (CDK bond order) first, then its
    /// implicit hydrogens.
    fn nbrs(&self, a: usize) -> impl Iterator<Item = Nb> + '_ {
        let m = &self.v.m;
        m.adj[a]
            .iter()
            .map(move |&(w, b)| {
                (
                    Some(w as usize),
                    m.order[b as usize],
                    self.arom_bond[b as usize],
                )
            })
            .chain(std::iter::repeat_n(
                (None, 1u8, false),
                m.hcount[a] as usize,
            ))
    }

    #[inline]
    fn z_of(&self, x: Option<usize>) -> u8 {
        x.map_or(1, |w| self.v.m.z[w])
    }

    #[inline]
    fn arom_of(&self, x: Option<usize>) -> bool {
        x.is_some_and(|w| self.arom[w])
    }

    fn calc(&self) -> f64 {
        let mut logp = 0.0;
        for a in 0..self.v.n() {
            let Some(c) = self.type_code(a).and_then(coefficient) else {
                return f64::NAN;
            };
            logp += c;
            let h = self.v.m.hcount[a] as usize;
            if h > 0 {
                let Some(c) = coefficient(self.hydrogen_code(Some(a), 0)) else {
                    return f64::NAN;
                };
                logp += c * h as f64;
            }
        }
        logp
    }

    /// `getAtomTypeCode` of a real atom (`None`: CDK's null code).
    fn type_code(&self, a: usize) -> Option<i32> {
        let m = &self.v.m;
        let z = m.z[a];
        if z == 1 {
            let nb = m.adj[a].first().map(|&(w, _)| w as usize);
            return Some(self.hydrogen_code(nb, m.charge[a]));
        }
        let non_h = m.adj[a]
            .iter()
            .filter(|&&(w, _)| m.z[w as usize] != 1)
            .count() as i32;
        let toadd = match z {
            6 => self.carbon_special(a),
            7 => self.nitrogen_special(a),
            8 => self.oxygen_special(a),
            9 => self.fluorine_special(a),
            _ => self.default_special(a),
        };
        if toadd == 99 {
            return None;
        }
        Some(100_000 * (m.charge[a] as i32 + 1) + z as i32 * 1000 + non_h * 100 + toadd)
    }

    /// Code of a hydrogen bonded to `x` (`None`: no bond).
    fn hydrogen_code(&self, x: Option<usize>, charge: i8) -> i32 {
        let non_h = match x {
            Some(w) if self.v.m.z[w] != 1 => 1,
            _ => 0,
        };
        100_000 * (charge as i32 + 1) + 1000 + non_h * 100 + self.hydrogen_special(x)
    }

    fn hydrogen_special(&self, x: Option<usize>) -> i32 {
        let Some(c) = x else { return 0 };
        if self.v.m.z[c] != 6 {
            return 50;
        }
        if self.carbonyl_conjugated(c) {
            return 51;
        }
        let ox = self.num_more_electronegative(c);
        match self.v.degree_all(c) {
            4 => match ox {
                0 => 46,
                1 => 47,
                2 => 48,
                3 => 49,
                _ => 0,
            },
            3 => match ox {
                0 => 47,
                1 => 48,
                _ => 49,
            },
            2 => {
                if ox == 0 {
                    48
                } else {
                    49
                }
            }
            1 => 121,
            _ => 0,
        }
    }

    /// `getNumMoreElectronegativethanCarbon`: bond orders to N, O, S, halogens.
    fn num_more_electronegative(&self, a: usize) -> u32 {
        self.nbrs(a)
            .filter(|&(w, _, _)| electron_withdrawing(self.z_of(w)))
            .map(|(_, o, _)| o as u32)
            .sum()
    }

    /// `getPolarBondArray`: [single, aromatic, double, triple] bonds to N, O, P, S.
    fn polar_bonds(&self, a: usize) -> [i32; 4] {
        let mut arr = [0; 4];
        for (w, o, ar) in self.nbrs(a) {
            if is_polar(self.z_of(w)) {
                if ar {
                    arr[1] += 1;
                } else {
                    match o {
                        1 => arr[0] += 1,
                        2 => arr[2] += 1,
                        3 => arr[3] += 1,
                        _ => {}
                    }
                }
            }
        }
        arr
    }

    fn double_bond_hetero(&self, a: usize) -> bool {
        self.nbrs(a)
            .any(|(w, o, ar)| !ar && is_polar(self.z_of(w)) && o == 2)
    }

    fn carbonyl_conjugated(&self, a: usize) -> bool {
        self.nbrs(a)
            .any(|(w, o, ar)| !ar && o == 1 && w.is_some_and(|w| self.double_bond_hetero(w)))
    }

    fn next_to_aromatic(&self, a: usize) -> bool {
        !self.arom[a] && self.nbrs(a).any(|(w, o, _)| self.arom_of(w) && o == 1)
    }

    fn bound_to(&self, a: usize, z: u8) -> bool {
        self.nbrs(a).any(|(w, _, _)| self.z_of(w) == z)
    }

    /// `checkAlphaCarbonyl`: a neighbour has a single bond to element `z`
    /// (the atom itself included).
    fn alpha_carbonyl(&self, a: usize, z: u8) -> bool {
        for (w, _, _) in self.nbrs(a) {
            match w {
                // an implicit hydrogen's only bond leads back to `a`
                None => {
                    if self.v.m.z[a] == z {
                        return true;
                    }
                }
                Some(w) => {
                    if self.nbrs(w).any(|(x, o, _)| self.z_of(x) == z && o == 1) {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn default_special(&self, a: usize) -> i32 {
        if self.arom[a] {
            10
        } else {
            self.polar_bonds(a).iter().sum()
        }
    }

    fn fluorine_special(&self, a: usize) -> i32 {
        if self.v.degree_all(a) != 1 {
            return 99;
        }
        let Some((w, _, _)) = self.nbrs(a).next() else {
            return 99;
        };
        match self.z_of(w) {
            16 => 8,
            5 => 9,
            6 => {
                let w = w.expect("carbon neighbour");
                let ox = self.num_more_electronegative(w);
                match self.v.degree_all(w) {
                    2 => 2,
                    3 => 3,
                    4 if ox <= 2 => 5,
                    4 => 7,
                    _ => 99,
                }
            }
            _ => 1,
        }
    }

    fn oxygen_special(&self, a: usize) -> i32 {
        match self.v.degree_all(a) {
            2 => {
                if self.bound_to(a, 7) {
                    1
                } else if self.bound_to(a, 16) {
                    2
                } else if self.arom[a] {
                    8
                } else {
                    3
                }
            }
            1 => {
                if self.bound_to(a, 7) {
                    4
                } else if self.bound_to(a, 16) {
                    5
                } else if self.alpha_carbonyl(a, 8) {
                    6
                } else if self.alpha_carbonyl(a, 7) {
                    9
                } else if self.alpha_carbonyl(a, 16) {
                    10
                } else {
                    7
                }
            }
            _ => 0,
        }
    }

    fn nitrogen_special(&self, a: usize) -> i32 {
        match self.v.degree_all(a) {
            4 => 9,
            3 => {
                if self.next_to_aromatic(a) {
                    1
                } else if self.carbonyl_conjugated(a) {
                    2
                } else if self.double_bond_hetero(a) {
                    10
                } else if self.polar_bonds(a)[0] > 0 {
                    3
                } else {
                    4
                }
            }
            2 => {
                if self.arom[a] {
                    5
                } else if self.double_bond_hetero(a) {
                    6
                } else {
                    7
                }
            }
            1 => 8,
            _ => 0,
        }
    }

    fn carbon_special(&self, a: usize) -> i32 {
        let [single, aromatic, double, triple] = self.polar_bonds(a);
        match self.v.degree_all(a) {
            4 => {
                if single > 0 {
                    3
                } else {
                    2
                }
            }
            3 => {
                if self.arom[a] {
                    if aromatic >= 1 && single == 0 {
                        11
                    } else if aromatic == 0 && single == 1 {
                        5
                    } else if aromatic >= 1 && single == 1 {
                        13
                    } else {
                        4
                    }
                } else if double == 1 && single == 0 {
                    7
                } else if single >= 1 && double == 0 {
                    8
                } else if double == 1 && single >= 1 {
                    14
                } else {
                    6
                }
            }
            2 => {
                if triple == 1 && single == 0 {
                    12
                } else if triple == 0 && single == 1 {
                    10
                } else if triple == 1 && single == 1 {
                    15
                } else {
                    9
                }
            }
            _ => {
                if single > 0 || double > 0 || aromatic > 0 || triple > 0 {
                    1
                } else {
                    0
                }
            }
        }
    }
}

/// `JPlogPDescriptor.WEIGHTS_CURR`: (atom type code, coefficient), sorted by code.
pub(crate) const JPLOGP_WEIGHTS: [(i32, f64); 182] = [
    (5201, 0.2201279736215536),
    (5202, 0.19045980382858146),
    (5401, 0.10441747048024147),
    (5402, 0.05162677089603265),
    (5403, 0.6788714142147848),
    (7108, 0.2634019114789942),
    (7207, -0.1902718970453204),
    (8104, -0.24195926611016633),
    (8105, -0.4215369017701643),
    (8106, -0.2964579842443157),
    (101146, 1.3994445700193028),
    (101147, 1.2616122128062803),
    (101148, 1.2711079053599976),
    (101149, 1.3258747052968567),
    (101150, -0.2363827296335956),
    (101151, 1.2556350911435799),
    (105100, 0.36881886842575007),
    (105201, 0.06904337130830694),
    (105202, -0.0745914491562598),
    (105300, -0.174370011345452),
    (105301, -0.1621399782797916),
    (105302, -0.17755682989875274),
    (105303, -0.1432571497021001),
    (106102, -3.616478490407742),
    (106103, -3.767115237033892),
    (106106, -2.0565095206356196),
    (106107, -2.416958126564593),
    (106109, -0.929959267287108),
    (106112, -1.3012751020335893),
    (106202, -2.349643420766774),
    (106203, -2.6563042881537227),
    (106204, -0.953850490807928),
    (106206, -1.0109064714897187),
    (106207, -1.5339493033443787),
    (106208, -1.195325147036),
    (106209, 0.19490613802180773),
    (106210, 0.16163164657523407),
    (106211, -1.1770219185049515),
    (106212, -0.2512600932013654),
    (106214, -1.6084019086078098),
    (106215, -0.14610561930731214),
    (106302, -1.3023723757750838),
    (106303, -1.5239332211866237),
    (106304, 0.20930194677993302),
    (106305, 0.11154050989797104),
    (106306, 0.01713355115342431),
    (106307, -0.2690238196488019),
    (106308, 0.026068238656834854),
    (106311, 0.1427758223243856),
    (106313, 0.2897701159093737),
    (106314, -0.39178934954486583),
    (106402, -0.19123067665076543),
    (106403, -0.6035455702256183),
    (107101, 0.051702151541644426),
    (107102, 0.06496457627779738),
    (107103, -0.2691841786263712),
    (107104, -0.33802382979998147),
    (107106, 0.3859974866654674),
    (107107, 0.394978253857827),
    (107108, 0.11642255395641928),
    (107201, 0.25454958160787483),
    (107202, 0.15973823557158393),
    (107203, 0.20036994532453556),
    (107204, 0.1275311923081761),
    (107205, -0.2543417785988061),
    (107206, 0.3113299058370724),
    (107207, 0.3145047278208787),
    (107301, 0.1787473725640726),
    (107302, 0.17510323694483412),
    (107303, -0.016959741231455404),
    (107304, 0.1542614658530209),
    (108101, -0.09994092418236927),
    (108102, 0.09143879448168145),
    (108103, 0.009836423634389751),
    (108104, 0.015474674863268114),
    (108105, -0.4749180976677123),
    (108106, 0.19048034389371205),
    (108107, 0.03602987937616161),
    (108109, -0.10666832936758427),
    (108110, 0.08928780909185945),
    (108201, 0.030702954811754002),
    (108202, 0.12058552604519218),
    (108203, 0.16547595574733062),
    (108208, 0.06665724849079398),
    (109101, 0.41374364339440817),
    (109103, 0.26651823980406203),
    (109105, 0.43019405241170144),
    (109107, 0.46487210540519147),
    (109108, 0.2430918227919903),
    (109109, 0.4801727828186138),
    (114100, 0.3587262776601395),
    (114200, 0.01526633068977459),
    (114300, -0.3058393642193975),
    (114301, -0.13761744158191205),
    (114302, -0.3095457315739295),
    (114400, -0.11455385699777243),
    (114401, -0.16389842380630032),
    (114402, -0.1103226433176957),
    (114403, -0.1553108949228919),
    (114404, -0.13111265719465162),
    (115100, 0.6096777283013224),
    (115200, 0.3379378715836258),
    (115201, 0.09994758075256505),
    (115202, 0.30788026393512663),
    (115210, -0.08346526510422402),
    (115300, 0.08797456644925557),
    (115301, -0.023836916386805847),
    (115302, -0.10814292962539623),
    (115303, -0.23605983536956895),
    (115400, -0.10861521692106911),
    (115401, -0.33332080419904164),
    (115402, -0.37912343134016635),
    (115403, -0.7356011823089021),
    (115404, -0.8103937160471404),
    (115500, 0.17750686328369028),
    (115501, -0.14068629542864905),
    (115502, -0.11992335751384754),
    (115503, 0.013887172778612027),
    (115504, -0.1025811855926768),
    (115505, -0.34311166776744884),
    (116100, 0.7001698255404422),
    (116101, 1.2551746199963494),
    (116200, 0.7928741182178681),
    (116201, 0.6390448917281739),
    (116202, 0.5335850658686033),
    (116210, 0.5791544003852439),
    (116300, 0.7367961788572792),
    (116301, -0.04914876490157527),
    (116302, -0.12643665926860584),
    (116303, -0.16169601215900375),
    (116400, 0.9078902724309305),
    (116401, 0.7460081218102875),
    (116402, 0.5536499705080733),
    (116403, 0.3132146936243478),
    (116404, 0.0372737885433136),
    (116503, 0.4425591506257104),
    (116600, 1.2424471654297324),
    (117100, 0.7116079866622599),
    (117101, 0.7369857420763359),
    (133200, -0.3076946592117401),
    (133201, -0.49680932374826),
    (133300, -0.2042709645502799),
    (133301, 0.25473886515880656),
    (133302, -0.7030388263360682),
    (133303, -0.4317778955330324),
    (133401, -0.3639701318790265),
    (133402, -0.2557512835716269),
    (133403, -0.8169297847709985),
    (133404, -0.6891007636859257),
    (134100, -0.1363315394071071),
    (134101, 0.1092440266426836),
    (134200, -0.04910253697266963),
    (134201, 0.011171177597208612),
    (134202, -0.631693669606887),
    (134210, 0.8725452294362313),
    (134300, -0.4657894122488925),
    (134301, -0.3795150596315356),
    (134302, -0.37554769340770927),
    (134303, -0.36036185997589393),
    (134400, -0.601185704550091),
    (134404, -1.1724223083800398),
    (135100, 0.8603892414982898),
    (135101, 0.7235643505309818),
    (153100, 0.9270429695306335),
    (153101, 1.0145354986151272),
    (153200, 0.7888787630537003),
    (153301, 0.5748646504612611),
    (153302, 0.6015475892374368),
    (207110, -0.26496784659823264),
    (207206, -0.24558051744952064),
    (207207, -0.07239795705976523),
    (207301, 0.36070459894156437),
    (207302, 0.24535814336356004),
    (207303, -0.47290811359153156),
    (207304, -0.07464529856367844),
    (207310, 0.35707776316923207),
    (207409, -0.23852255872700964),
    (208208, 0.7099234399396344),
    (216210, 0.7113783097640652),
    (216300, 0.8104414861979382),
    (216301, 0.16768140357444056),
    (216310, 0.16982252294512776),
];
