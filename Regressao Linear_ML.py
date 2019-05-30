# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:34:23 2019

@author: Wayner
"""

import numpy as np

def regressao_linear_treino (X, y):
    XT = np.transpose(X.copy())
    XTX = np.matmul(XT,X)
    A = np.matmul(np.linalg.inv(XTX),XT)
    return np.matmul(A,y)

def regressao_linear (X, w):
    return np.matmul(X, w)



dataset = np.array([
        [2.45663631187698, -18.7035937633755],
        [1.31244205850455, 7.76904667956506],
        [1.54701339111928, 87.6118074125951],
        [2.65218673258594, -1.43779585941208],
        [3.76183410715101, 32.9742497024237],
        [4.65946360163057, -64.4682528228529],
        [5.01774595843772, -41.9860256222228],
        [4.7015172047513, 95.9703278859523],
        [5.32728003249194, 10.9881958192957],
        [6.34816745035322, 98.4119419531101],
        [6.63781962849453, -40.400813341354],
        [7.87247652435596, 18.8709256070807],
        [7.90491902288844, -56.8548881031843],
        [7.68942680530775, 90.9042555277954],
        [7.96080393372312, -13.2871255466715],
        [9.49218383137032, 120.978566658915],
        [9.68047480900735, 116.054428957593],
        [10.3874571431352, -57.7528133854607],
        [10.3418806940364, -23.0996444250125],
        [10.2678859545795, -49.8016972426649],
        [12.020983918929, 41.9586190959379],
        [11.4093340143927, 101.397914190313],
        [13.3035632088643, -42.154680435756],
        [13.2181954775262, 27.7493184683262],
        [13.0881362242057, 20.5665864208259],
        [13.7178479664656, -28.4015568249461],
        [14.1354678128857, 17.6319213132721],
        [14.0737575140604, 26.6761291844061],
        [16.2357188005838, 2.38714099122329],
        [15.4046512756738, -33.469514970992],
        [16.1824328316035, 125.436753399839],
        [16.0146736409334, 22.9398290967887],
        [17.5472880362255, -29.4433448163365],
        [18.4672342548408, 13.8636391592817],
        [18.3088138592074, -41.9710149377956],
        [18.465778089677, 79.380844276728],
        [20.4438924070186, 8.791436495923],
        [20.546902648101, 132.334358072691],
        [21.4506021385346, -33.9084740218453],
        [21.1981292187711, 48.5905532029991],
        [21.906828376056, 41.3221718106233],
        [21.9814658591365, -29.6012974854022],
        [21.6176324772645, 102.420075265629],
        [23.9423947935003, 134.891140066869],
        [23.0765265797177, 27.435913857539],
        [23.3822686319092, 123.091210158151],
        [25.1183831926714, 105.013863249399],
        [24.6172359984342, -18.5811909333224],
        [26.259227425567, 28.2549854470844],
        [26.9054088207802, 139.313203141558],
        [26.6940530689147, 62.5443418125176],
        [26.177170982292, 37.209918058502],
        [28.155284969191, 142.770262642516],
        [27.2143407412006, 152.476547058862],
        [28.1165576446716, 130.817776662125],
        [28.6136691329409, 45.8896442192293],
        [28.7958858352865, -6.42397961633706],
        [29.469143881659, 35.9170357351536],
        [30.9099453602127, -7.92338661733946],
        [31.1521182008882, 106.954392720643],
        [32.2916672163022, 160.955849504602],
        [31.6295978460485, -5.25487820624726],
        [33.2755225241484, 76.2067989197454],
        [33.8259487393022, 64.2304006919927],
        [33.9639204429314, 198.071240202526],
        [33.663759010316, 118.367252624562],
        [34.9776768896063, 59.3692472381339],
        [34.6859415692388, 135.014176764008],
        [34.8563642076841, 39.0277748114563],
        [36.7422608048247, 180.043085095044],
        [36.1357752350512, 58.407257290954],
        [37.1268117261889, 66.7654613628112],
        [37.5403710466106, 115.09329648604],
        [37.7367956213625, 209.781202171042],
        [39.2520132543367, 157.210346177156],
        [39.8989726321184, 210.363482966024],
        [39.7793172012504, 189.92010877801],
        [39.1635275160817, 108.332009262612],
        [40.2406998032897, 159.652897001744],
        [41.6784602705337, 206.38949620823],
        [41.6319282597472, 219.477728383097],
        [41.2654216153966, 43.4351403824105],
        [42.6534705026839, 36.8985149769015],
        [43.8063930754601, 79.6804282148827],
        [42.9243273306328, 61.3457605779786],
        [43.1001871864283, 64.8698203319576],
        [43.9001709054699, 161.073563561367],
        [45.8855945846802, 74.3263943599249],
        [45.3092811348743, 151.801811128949],
        [46.5979960027085, 116.329321471928],
        [46.3738397708193, 95.6628841212356],
        [46.7117600200888, 128.765284560253],
        [47.9443455346259, 205.037194105602],
        [48.9024027518865, 73.1529017672265],
        [48.6318716509691, 49.6840530403988],
        [49.307587381037, 206.489212464133],
        [49.1770897357243, 145.367709346827],
        [49.5019709646404, 118.243599152015],
        [49.7136378977308, 182.448788100181],
        [50.1990311403869, 97.0154353523808],
        [51.8922182727461, 192.821053250751],
        [51.9229325677509, 78.4512394116552],
        [52.9479160397175, 187.598141272872],
        [53.4896689024969, 92.3796480228662],
        [54.3017105473286, 159.947744153825],
        [53.5308790665984, 191.555416118211],
        [54.0006391751187, 161.69739173625],
        [55.3482531607351, 68.5324235130035],
        [54.7815973733506, 202.12366911349],
        [55.9216962941731, 264.681245336274],
        [57.1456276446917, 81.2036996874725],
        [57.3438813685084, 98.4431564339959],
        [57.7163099834241, 232.706044160659],
        [57.4994382338376, 249.591965270789],
        [58.8014601003817, 221.919524469756],
        [59.3712448262669, 274.227146508964],
        [59.1488138210595, 181.983849662496],
        [60.3990437875184, 106.302973285696],
        [60.1166070564456, 118.113869527221],
        [61.1056998258984, 204.806988053118],
        [61.7953167025082, 223.015678700147],
        [61.7963773493434, 165.646972270679],
        [62.1504285897449, 111.92025510464],
        [63.3545527709332, 135.187819390583],
        [62.6871718466287, 131.696500725447],
        [64.1420207469558, 145.567932375978],
        [65.1857288768345, 111.775266714689],
        [65.0361851992338, 163.180570485935],
        [66.0487940981559, 167.148434214084],
        [65.1541420949867, 113.220050188325],
        [65.8677290435521, 151.464690576387],
        [66.4298821222652, 152.599097322817],
        [66.848385159071, 111.493859279593],
        [68.5468753415179, 257.901476283487],
        [67.5448078378694, 213.566822937834],
        [68.1034550571086, 164.530877489493],
        [69.7671103764346, 252.86655440855],
        [70.6358979558602, 204.611036123866],
        [71.4618381112909, 165.685327222045],
        [71.9286846913465, 292.258354596296],
        [72.2083091174083, 135.902916675892],
        [72.9254344118974, 299.23063695116],
        [72.5981550503912, 162.298968147822],
        [73.5243174822166, 155.869323268476],
        [73.3686153284545, 192.933563052027],
        [73.9825527913087, 188.988172065721],
        [74.2821964470049, 215.223112784938],
        [74.4865119530858, 138.333022256987],
        [74.738547195205, 258.414828835835],
        [75.8752565892796, 252.801533947658],
        [76.9111858360313, 200.484690420356],
        [77.9911739578107, 327.221182416714],
        [76.6462267417051, 250.785694396024],
        [77.8576201462216, 305.533504393845],
        [78.2577091339542, 184.185467222239],
        [78.1794807138378, 140.114757308677],
        [79.7729385090845, 293.749341340862],
        [79.2581305684732, 134.53936912363],
        [81.1052505455737, 211.936830062517],
        [81.4656480203219, 236.520631101502],
        [81.3704770260042, 299.702532010328],
        [81.3540857747222, 191.741571833983],
        [82.9225160191371, 326.503780040097],
        [82.5478875598301, 196.333956209271],
        [84.3081108189799, 154.524100812279],
        [84.5120700906364, 152.049166994815],
        [84.7586921893272, 227.225230072746],
        [85.0234509863993, 295.214621809774],
        [84.6047159740219, 313.590381249801],
        [85.3856194766872, 266.913065091011],
        [86.9563637713253, 210.000506548756],
        [87.2705045757917, 214.528718594224],
        [88.48553376456, 234.320726842557],
        [88.9072455012959, 355.843203370407],
        [88.2473058717927, 320.272123768342],
        [89.6071821236139, 248.23192013386],
        [89.330989944693, 223.022160929345],
        [89.2167579784142, 341.817984697508],
        [89.7637192088299, 205.503320581597],
        [90.3950906027198, 237.566661755208],
        [91.2865057461094, 274.753520332857],
        [92.4969869932042, 262.421584861774],
        [93.3372704623858, 198.827640630952],
        [93.3932627863107, 197.084127778281],
        [94.3307402300061, 309.305492348228],
        [93.8795412011679, 256.415851192717],
        [93.9645241816766, 210.630871613794],
        [94.460228282358, 241.718232087376],
        [94.504365381, 203.893064463549],
        [96.3574354434527, 241.773029159992],
        [95.6469011088, 356.454127968298],
        [96.247516724064, 232.764708668005],
        [97.4264703044939, 382.602693722948],
        [98.9808089704645, 202.549845888411],
        [99.1931848603543, 296.921809850539],
        [99.0049103359252, 371.26117673072],
        [99.0296527853449, 222.171695472675],
        [100.257292298868, 367.902106791302],
        [100.128660235932, 356.449402259592],
        [100.99324483951, 392.660385744318],
        [101.315843628005, 237.065897170163],
        [101.140667742791, 227.706084896085],
        [103.333687722819, 391.062845869254],
        [102.854694747758, 263.767426117951],
        [103.615790777163, 283.107355771892],
        [103.229558988774, 244.69179990286],
        [104.266273464137, 249.121426104983],
        [105.380535063412, 238.489787504422],
        [105.064835494897, 291.094874162896],
        [105.925849477598, 256.4851044036],
        [106.057157782065, 401.177360483548],
        [106.591989136749, 398.996496290328],
        [107.560859980232, 253.789887375391],
        [108.985204335219, 254.975014541138],
        [109.438488012956, 417.392127037069],
        [109.440583406465, 405.536395630368],
        [108.665239297516, 273.668717219462],
        [110.589458848816, 423.226454579402],
        [109.738370197273, 339.2429708748],
        [111.298500662308, 286.844832297717],
        [111.621334445533, 257.190757598216],
        [112.599358622319, 364.581356002799],
        [112.518796567341, 326.119542680428],
        [112.42404176653, 293.209744849217],
        [112.82448320413, 365.758912660674],
        [114.391757526077, 360.541947738723],
        [114.554794967813, 256.885375576367],
        [115.986511507261, 397.379015899337],
        [114.990754401343, 378.793684821981],
        [115.052776986238, 378.887259554281],
        [117.09650710037, 337.857829947965],
        [117.06290899867, 247.91299170307],
        [117.836039064074, 427.628585191787],
        [117.981019107243, 357.551100166772],
        [119.435752388743, 284.897055847272],
        [119.770309142059, 276.803209694586],
        [120.082736928485, 452.923791309632],
        [119.453740143479, 417.281417021063],
        [120.137172225753, 433.552193723184],
        [120.112179375393, 297.851511669518],
        [120.769512917281, 417.664206951762],
        [121.787605149651, 356.934658613315],
        [123.493864260115, 403.787417351918],
        [122.271453401503, 409.256227622334],
        [123.660769533838, 449.402730889532],
        [123.290057892761, 427.820015244779],
        [125.187614609549, 321.574462088108],
        [124.681775924899, 400.782858982882],
        [125.998445485895, 413.361795923164],
        [126.77336868893, 423.920651540075],
        [126.405717178794, 379.518693718252],
        [126.106613731279, 330.163048661076],
        [127.463361276972, 283.698976229043],
        [127.110029449614, 315.25749174518],
        [129.038601217887, 426.068333669738],
        [128.720612848934, 432.724683984837],
        [130.388879453565, 301.197233639979],
        [130.965073257632, 347.937296153364],
        [131.274323631091, 407.735638723372],
        [131.377658084504, 434.385002483214],
        [132.197821985153, 469.576334416844],
        [131.633412475748, 361.488859791975],
        [133.092077425119, 396.314762466202],
        [133.813628421458, 308.606753137362],
        [133.805034610372, 351.720777603288],
        [133.712960268843, 333.236695174835],
        [134.307972368356, 317.083575560514],
        [134.188905994213, 356.979185917742],
        [135.572578035133, 369.571825491123],
        [136.569496185044, 388.58875761753],
        [136.525982223682, 317.920981088887],
        [137.894880132718, 314.998146222611],
        [138.488947457815, 467.123236121418],
        [138.574928701168, 349.7487906878],
        [138.195343174823, 458.130421081029],
        [138.290669479625, 424.433099760785],
        [140.374057578989, 407.376568562143],
        [140.537514301881, 342.593381989604],
        [140.956116102412, 387.431441341987],
        [141.67687710642, 484.246344601194],
        [141.35982673693, 446.916188767593],
        [142.531978844152, 494.479426424364],
        [143.324290880002, 386.460019631063],
        [143.934751888791, 477.623419279502],
        [143.10699817727, 400.969804055103],
        [143.77421488846, 419.586546027624],
        [144.465085255866, 418.82138907381],
        [145.65565384154, 424.531605692021],
        [146.461558911995, 443.53303402362],
        [146.756445456676, 359.195543607553],
        [145.896222335159, 402.083999618151],
        [146.365905633089, 463.20872163918],
        [148.184321524683, 370.937412070096],
        [147.97634717151, 521.766515027145],
        [149.063447787642, 409.098710574478],
        [148.550339860245, 484.169431439645],
        [149.983305718536, 494.470445481392],
        [150.32810951581, 532.06128578243],
        [149.736906420379, 389.137536294836],
        [151.656971610145, 541.261182112315]])

trainset = dataset[0:300:6,:]

#trainset = np.array([[69.04, 139],
#             [61.04, 126],
#             [33.04, 90],
#             [81.04, 144],
#             [93.04, 163],
#             [54.04, 136],
#             [15.04, 61],
#             [64.04, 62],
#             [18.04, 41],
#             [45.04, 120]])

    
#trainset = np.array([[-26,52, 139, 0.115],
#                     [-34,52, 126, 0.12],
#                     [-62,52, 90, 0.105],
#                     [-14,52, 144, 0.090],
#                     [-2,52, 163, 0.100],
#                     [-41,52, 136, 0.120],
#                     [-80,52, 61, 0.105],
#                     [-31,52, 62, 0.080],
#                     [-77,52, 41, 0.100],
#                     [-50,52, 120, 0.115]])


#trainset = np.array([[122, 1, 139, 0.115],
#                     [114, 1, 126, 0.12],
#                     [86, 1, 90, 0.105],
#                     [134, 1, 144, 0.090],
#                     [146, 1, 163, 0.100],
#                     [107, 1, 136, 0.120],
#                     [68, 1, 61, 0.105],
#                     [117, 1, 62, 0.080],
#                     [71, 1, 41, 0.100],
#                     [98, 1, 120, 0.115]])
    

w = regressao_linear_treino(trainset[:,0:(len(trainset[0,:])-1)], trainset[:,len(trainset[0,:])-1])

ytrain = regressao_linear (trainset[:,0:(len(trainset[0,:])-1)], w)
print(ytrain-trainset[:,(len(trainset[0,:])-1)])

yaprox = w*dataset[:,0]
yreal = 2.94736264847*dataset[:,0]