import numpy as np

losses = [2.62257e-05, 1.2278481e-05, 1.3828182e-05, 1.3232144e-05, 0.010375239, 0.012858976, 2.0384581e-05, 1.180165e-05, 2.5868081e-05, 6.866219e-05, 1.001353e-05, 1.633154e-05, 0.00025090406, 1.20400655e-05, 1.4305013e-05, 1.239769e-05, 1.0967195e-05, 1.20400655e-05, 78.37992, 1.1563234e-05, 1.1086402e-05, 1.1086402e-05, 1.4781843e-05, 3.1470758e-05, 1.0490363e-05, 2.491443e-05, 1.2993728e-05, 1.490105e-05, 0.00010966653, 9.7751135e-06, 1.537788e-05, 2.62257e-05, 1.0251946e-05, 1.0251946e-05, 0.000735968, 0.0021111872, 1.1920858e-05, 1.2993728e-05, 1.001353e-05, 1.001353e-05, 1.7404405e-05, 1.2159274e-05, 1.1563234e-05, 0.0004355802, 0.00023517228, 1.0967195e-05, 9.1790735e-06, 9.536698e-06, 3.8980677e-05, 1.0132739e-05, 1.12056105e-05, 1.1682442e-05, 0.00016866691, 1.1920858e-05, 1.3470559e-05, 0.013698876, 1.2278481e-05, 6.270212e-05, 1.12056105e-05, 1.0251946e-05, 79.46151, 2.9325056e-05, 1.2159274e-05, 1.5973917e-05, 1.4305013e-05, 1.1086402e-05, 0.0095050195, 0.00044630852, 1.1324818e-05, 1.3828182e-05, 1.1682442e-05, 1.1563234e-05, 0.0016370715, 5.2927524e-05, 1.060957e-05, 1.2993728e-05, 1.5020258e-05, 1.060957e-05, 1.4305013e-05, 2.324554e-05, 1.1086402e-05, 1.28745205e-05, 1.2755313e-05, 1.6927575e-05, 1.20400655e-05, 1.41858045e-05, 1.1682442e-05, 3.373566e-05, 0.0007815882, 1.4781843e-05, 1.5973917e-05, 1.180165e-05, 1.4305013e-05, 1.12056105e-05, 0.0035601964, 1.633154e-05, 1.3589766e-05, 0.00047420204, 1.2278481e-05, 1.716599e-05, 1.1086402e-05, 9.44093e-05, 6.544376e-05, 1.4543428e-05, 0.00108462, 4.1722382e-05, 9.061596, 1.5258673e-05, 1.4305013e-05, 0.00093595084, 1.0967195e-05, 1.001353e-05, 1.41858045e-05, 0.0001982254, 1.180165e-05, 1.12056105e-05, 2.0265374e-05, 4.9351427e-05, 1.180165e-05, 1.12056105e-05, 9.894322e-06, 1.1682442e-05, 1.2755313e-05, 1.3351351e-05, 1.4305013e-05, 1.239769e-05, 1.0251946e-05, 1.4543428e-05, 1.394739e-05, 1.2755313e-05, 0.0014472037, 1.6212332e-05, 1.7285198e-05, 1.0847986e-05, 1.0847986e-05, 1.0967195e-05, 0.00018249277, 1.239769e-05, 1.1324818e-05, 0.0074242637, 1.633154e-05, 1.239769e-05, 1.394739e-05, 1.2159274e-05, 1.0490363e-05, 2.7894584e-05, 1.0728779e-05, 2.9682673e-05, 0.0048438385, 1.5020258e-05, 1.4305013e-05, 5.674201e-05, 1.0847986e-05, 7.251156, 3.349725e-05, 2.2768714e-05, 1.6569955e-05, 1.5258673e-05, 0.001894513, 1.060957e-05, 1.0251946e-05, 1.001353e-05, 2.5868081e-05, 1.3708975e-05, 1.2755313e-05, 0.00058895885, 1.6450747e-05, 1.9430925e-05, 2.4437606e-05, 1.28745205e-05, 1.6450747e-05, 2.3603161e-05, 1.3828182e-05, 0.0009279965, 0.0012337315, 1.001353e-05, 1.3589766e-05, 1.2636105e-05, 1.1682442e-05, 1.1920858e-05, 5.9960475e-05, 1.1682442e-05, 5.1377883e-05, 0.00039481252, 0.000113480804, 3.1351552e-05, 2.0265374e-05, 1.394739e-05, 6.8791833, 1.394739e-05, 2.9563467e-05, 1.7523613e-05, 1.3470559e-05, 2.8967437e-05, 1.2993728e-05, 1.0847986e-05, 9.655906e-06, 2.527205e-05, 3.981511e-05, 0.0007617151, 9.4335575, 1.03711545e-05, 1.03711545e-05, 0.00056753756, 2.62257e-05, 1.239769e-05, 1.8119648e-05, 4.148397e-05, 0.9026696, 1.5973917e-05, 1.2516897e-05, 2.3364748e-05, 0.0002124084, 0.0038152777, 1.7523613e-05, 1.0967195e-05, 1.490105e-05, 2.2768714e-05, 1.394739e-05, 1.12056105e-05, 1.1920858e-05, 0.001966806, 2.0861407e-05, 1.2636105e-05, 1.2993728e-05, 3.850386e-05, 1.1444026e-05, 1.5139465e-05, 2.5748875e-05, 1.2278481e-05, 7.378783e-05, 1.3589766e-05, 1.1563234e-05, 2.6344906e-05, 1.3351351e-05, 0.00046132813, 1.3232144e-05, 1.180165e-05, 77.31535, 1.8238856e-05, 2.6344906e-05, 1.0728779e-05, 2.491443e-05, 1.6927575e-05, 1.9430925e-05, 4.2437605e-05, 8.239659, 2.3126335e-05, 0.00012909532, 1.3351351e-05, 1.4066597e-05, 1.716599e-05, 1.180165e-05, 2.0980615e-05, 9.7751135e-06, 2.002696e-05, 1.1920858e-05, 0.0002826053, 2.62257e-05, 1.3351351e-05, 1.3708975e-05, 0.22663471, 1.1324818e-05, 1.12056105e-05, 0.0020782794, 2.801379e-05, 1.12056105e-05, 0.00094086974, 1.28745205e-05, 1.1682442e-05, 1.3828182e-05, 5.161629e-05, 1.2516897e-05, 1.3351351e-05, 0.0051837536, 2.4556812e-05, 2.3722367e-05, 1.0967195e-05, 1.9192512e-05, 1.2755313e-05, 1.883489e-05, 1.394739e-05, 1.3351351e-05, 5.125868e-05, 7.2717366, 0.0004688379, 1.1086402e-05, 1.3232144e-05, 1.0490363e-05, 1.001353e-05, 0.0036481377, 1.7404405e-05, 1.5735503e-05, 1.28745205e-05, 0.0036556448, 0.0054650903, 1.2516897e-05, 3.6000558e-05, 5.09209, 0.00042377904, 1.1444026e-05, 0.0004165497, 1.7404405e-05, 1.1086402e-05, 0.00015150353, 0.00079266593, 1.060957e-05, 1.180165e-05, 1.394739e-05, 1.5497088e-05, 1.2755313e-05, 2.741776e-05, 1.442422e-05, 1.6927575e-05, 0.012168457, 1.3589766e-05, 1.6808368e-05, 6.472855e-05, 1.060957e-05, 1.3232144e-05, 0.001093202, 1.03711545e-05, 9.417489e-06, 1.12056105e-05, 1.12056105e-05, 1.7523613e-05, 1.1324818e-05, 1.6093125e-05, 1.3112935e-05, 1.001353e-05, 1.7762026e-05, 1.41858045e-05, 1.5139465e-05, 1.9073304e-05, 3.8146245e-05, 1.3589766e-05, 0.1687174, 2.7656173e-05, 1.4305013e-05, 1.6450747e-05, 6.385473, 0.031220421, 1.3351351e-05, 0.0096157715, 9.5854845, 2.002696e-05, 1.0490363e-05, 91.46862, 1.060957e-05, 1.8715684e-05, 1.2755313e-05, 2.2887922e-05, 1.2993728e-05, 2.3126335e-05, 1.7285198e-05, 0.003381094, 1.6927575e-05, 1.0967195e-05, 1.0132739e-05, 1.180165e-05, 3.7311813e-05, 1.3232144e-05, 1.5020258e-05, 2.3483954e-05, 2.193427e-05, 1.8119648e-05, 1.239769e-05, 1.2755313e-05, 1.1682442e-05, 1.1920858e-05, 1.3828182e-05, 1.5020258e-05, 1.2636105e-05, 1.2755313e-05, 0.0045006936, 1.3589766e-05, 1.1682442e-05, 1.1444026e-05, 1.1086402e-05, 2.1457441e-05, 4.9828242e-05, 2.5033638e-05, 1.2278481e-05, 4.3033626e-05, 1.6093125e-05, 0.0014246766, 1.1324818e-05, 1.5258673e-05, 7.9540825, 1.7404405e-05, 0.005144079, 3.5404533e-05, 1.490105e-05, 1.5616295e-05, 1.5735503e-05, 1.6569955e-05, 1.1444026e-05, 9.417489e-06, 1.1086402e-05, 1.5258673e-05, 0.00374573, 8.65422e-05, 2.0622994e-05, 36.53431, 1.2278481e-05, 1.5735503e-05, 0.0008623457, 1.1563234e-05, 8.785339e-05, 7.211902e-05, 1.0847986e-05, 1.2755313e-05, 2.467602e-05, 1.442422e-05, 1.0728779e-05, 1.1324818e-05, 1.1563234e-05, 1.9550132e-05, 1.1563234e-05, 2.2411095e-05, 1.2516897e-05, 1.0847986e-05, 74.98435, 1.9430925e-05, 3.087473e-05, 1.4305013e-05, 1.3589766e-05, 1.0251946e-05, 1.1444026e-05, 1.490105e-05, 1.6808368e-05, 1.2278481e-05, 1.4543428e-05, 1.0967195e-05, 1.800044e-05, 1.5139465e-05, 98.32515, 1.180165e-05, 0.013075209, 2.5391257e-05, 1.4543428e-05, 1.4066597e-05, 8.695099, 1.0251946e-05, 1.3351351e-05, 1.0967195e-05, 7.903264e-05, 1.239769e-05, 0.0010620921, 1.3708975e-05, 1.1682442e-05, 1.12056105e-05, 2.6702524e-05, 1.1324818e-05, 1.7642818e-05, 1.0251946e-05, 1.3232144e-05, 1.800044e-05, 0.00096232514, 2.0265374e-05, 2.2291888e-05, 8.061954, 1.1086402e-05, 1.3112935e-05, 1.1920858e-05, 1.12056105e-05, 9.894322e-06, 1.6093125e-05, 0.00051830686, 0.00036691874, 0.00017975146, 1.490105e-05, 9.655906e-06, 3.1828375e-05, 1.7762026e-05, 2.0384581e-05, 4.7205765e-05, 1.585471e-05, 0.00066234585, 1.5735503e-05, 1.1682442e-05, 0.00031752314, 2.6583319e-05, 1.1682442e-05, 1.0728779e-05, 1.0132739e-05, 1.4781843e-05, 3.349725e-05, 6.1033294e-05, 6.258292e-05, 1.2159274e-05, 1.5735503e-05, 0.00081320904, 1.4662635e-05, 1.3828182e-05, 1.5497088e-05, 1.1920858e-05, 1.0847986e-05, 1.3470559e-05, 1.20400655e-05, 0.00029631038, 7.438383e-05, 6.556296e-05, 1.3470559e-05, 0.009409633, 1.1086402e-05, 4.088795e-05, 1.3351351e-05, 1.060957e-05, 2.1815062e-05, 3.4093275e-05, 1.5735503e-05, 1.8954097e-05, 3.492771e-05, 1.6808368e-05, 9.7751135e-06, 1.0728779e-05, 0.00042882306, 3.111314e-05, 1.1324818e-05, 2.741776e-05, 1.1920858e-05, 1.537788e-05, 1.2636105e-05, 2.6106494e-05, 4.005352e-05, 1.6212332e-05, 1.4305013e-05, 1.4066597e-05, 1.0967195e-05, 1.0490363e-05, 9.536698e-06, 1.4662635e-05, 1.2993728e-05, 1.2636105e-05, 1.0847986e-05, 0.00042699755, 1.2159274e-05, 1.0251946e-05, 7.7114906, 8.867996, 2.1815062e-05, 1.03711545e-05, 0.00023255029, 1.1563234e-05, 4.2914424e-05, 2.8848232e-05, 1.5616295e-05, 9.393251e-05, 2.2649509e-05, 2.1338235e-05, 3.170917e-05, 1.4662635e-05, 1.716599e-05, 1.6927575e-05, 3.3402114, 1.3470559e-05, 2.324554e-05, 1.1682442e-05, 2.6344906e-05, 7.0211805e-05, 2.3126335e-05, 1.3828182e-05, 0.08794473, 24.774874, 2.2053475e-05, 1.1444026e-05, 7.390703e-05, 0.00011097769, 1.3470559e-05, 3.266281e-05, 1.1682442e-05, 1.3828182e-05, 2.1099822e-05, 1.2278481e-05, 2.5868081e-05, 1.2636105e-05, 3.3616456e-05, 1.442422e-05, 1.001353e-05, 1.0251946e-05, 2.5391257e-05, 1.8119648e-05, 0.00072309445, 0.00057075603, 1.0847986e-05, 5.805324e-05, 1.9192512e-05, 2.3364748e-05, 1.2278481e-05, 1.239769e-05, 2.491443e-05, 1.1682442e-05, 3.552374e-05, 1.001353e-05, 1.2516897e-05, 8.8744755, 4.339124e-05, 1.9073304e-05, 2.3364748e-05, 0.0030571988, 1.239769e-05, 1.0251946e-05, 1.0728779e-05, 2.1099822e-05, 1.800044e-05, 60.983612, 1.1563234e-05, 1.8596476e-05, 0.00049245154, 1.9907753e-05, 1.5973917e-05, 1.0847986e-05, 1.490105e-05, 1.9311718e-05, 0.00042810812, 7.3987823, 1.0967195e-05, 0.00074455043, 1.03711545e-05, 2.476045, 0.00043343453, 1.1920858e-05, 1.5139465e-05, 9.417489e-06, 1.1682442e-05, 1.1682442e-05, 1.442422e-05, 0.00061688694, 1.1324818e-05, 1.3351351e-05, 9.7751135e-06, 2.3603161e-05, 1.3112935e-05, 0.0027472356, 1.490105e-05, 2.3483954e-05, 1.1444026e-05, 1.1444026e-05, 4.553691e-05, 1.1086402e-05, 1.1324818e-05, 1.0251946e-05, 1.6808368e-05, 1.7285198e-05, 2.0146166e-05, 3.7431015e-05, 1.5735503e-05, 1.1086402e-05, 0.0015866549, 1.7046783e-05, 1.0251946e-05, 1.6093125e-05, 1.4066597e-05, 1.3708975e-05, 1.3470559e-05, 1.966934e-05, 1.41858045e-05, 2.0861407e-05, 5.566919e-05, 93.35502, 1.2278481e-05, 0.0012780601, 1.239769e-05, 1.0967195e-05, 3.8384653e-05, 1.180165e-05, 1.633154e-05, 1.5497088e-05, 1.0847986e-05, 1.4066597e-05, 1.2516897e-05, 0.002204133, 2.2172682e-05, 1.001353e-05, 92.47556, 1.0251946e-05, 1.5735503e-05, 9.894322e-06, 9.7751135e-06, 0.001073142, 2.1695856e-05, 1.8358061e-05, 1.2159274e-05, 1.1563234e-05, 9.536698e-06, 7.6202426, 0.0006827169, 2.4795225e-05, 1.1444026e-05, 9.894322e-06, 9.655906e-06, 1.2159274e-05, 1.3708975e-05, 1.1086402e-05, 1.4543428e-05, 35.884106, 6.2225314e-05, 1.5020258e-05, 1.0490363e-05, 1.3589766e-05, 1.7881233e-05, 1.2993728e-05, 1.585471e-05, 6.210611e-05, 1.8954097e-05, 1.0490363e-05, 86.193245, 1.41858045e-05, 1.2755313e-05, 0.0017089413, 1.2159274e-05, 8.725185, 1.2636105e-05, 2.6702524e-05, 1.239769e-05, 1.180165e-05, 1.2516897e-05, 84.26548, 0.0027043333, 4.4225668e-05, 1.3708975e-05, 0.0009247781, 50.134697, 1.3589766e-05, 0.0006758075, 1.1324818e-05, 1.4543428e-05, 1.3708975e-05, 1.7404405e-05, 0.0007724431, 2.0622994e-05, 3.492771e-05, 1.633154e-05, 1.3351351e-05, 1.3708975e-05, 3.6834994e-05, 9.7751135e-06, 1.180165e-05, 4.3868058e-05, 2.0146166e-05, 1.8477269e-05, 1.4066597e-05, 6.837608, 1.716599e-05, 1.1682442e-05, 1.1324818e-05, 2.0384581e-05, 1.585471e-05, 1.41858045e-05, 0.008017973, 1.12056105e-05, 0.00061045005, 6.496695e-05, 1.239769e-05, 1.2516897e-05, 1.6450747e-05, 2.324554e-05, 1.2755313e-05, 2.0146166e-05, 2.8252203e-05, 1.0251946e-05, 0.004926406, 1.3589766e-05, 1.2516897e-05, 1.633154e-05, 1.3232144e-05, 2.467602e-05, 1.1682442e-05, 83.55306, 1.442422e-05, 1.03711545e-05, 1.2993728e-05, 1.5020258e-05, 6.842379e-05, 0.0065164813, 2.1695856e-05, 0.0010355116, 1.4781843e-05, 1.3828182e-05, 1.633154e-05, 1.0132739e-05, 1.1086402e-05, 6.842379e-05, 1.2755313e-05, 1.8715684e-05, 4.804019e-05, 1.585471e-05, 0.0041521746, 3.111314e-05, 1.1920858e-05, 9.1790735e-06, 79.25468, 1.2278481e-05, 1.2516897e-05, 2.0146166e-05, 0.00017998983, 1.03711545e-05, 1.0251946e-05, 1.668916e-05, 5.203074, 1.239769e-05, 2.0861407e-05, 1.1920858e-05, 1.6093125e-05, 73.5437, 1.3470559e-05, 1.1324818e-05, 7.331103e-05, 84.89117, 1.0132739e-05, 4.3748852e-05, 1.1563234e-05, 9.548208e-05, 8.49139, 99.741455, 1.2278481e-05, 1.394739e-05, 9.894322e-06, 1.883489e-05, 1.5258673e-05, 1.2159274e-05, 0.0075863805, 1.7404405e-05, 1.0132739e-05, 1.1563234e-05, 1.5616295e-05, 2.2411095e-05, 1.0967195e-05, 9.691246e-05, 1.3470559e-05, 6.89006e-05, 1.2278481e-05, 1.3708975e-05, 0.0013568728, 1.41858045e-05, 1.0728779e-05, 80.0766, 1.239769e-05, 94.22033, 1.2516897e-05, 1.585471e-05, 1.8477269e-05, 1.060957e-05, 1.4543428e-05, 8.932563, 1.716599e-05, 0.001235877, 2.5987287e-05, 1.9311718e-05, 1.7285198e-05, 1.1086402e-05, 2.1815062e-05, 1.3232144e-05, 8.9164576e-05, 1.03711545e-05, 0.00078944984, 1.2159274e-05, 0.00048720886, 1.4066597e-05, 1.3112935e-05, 7.827918, 0.00080355396, 1.060957e-05, 1.1086402e-05, 9.059865e-06, 1.0251946e-05, 2.491443e-05, 8.773419e-05, 0.0001463783, 1.1444026e-05, 0.0030516267, 2.3126335e-05, 1.180165e-05, 4.2795218e-05, 1.6569955e-05, 1.03711545e-05, 4.0449677, 1.2636105e-05, 1.6212332e-05, 61.57123, 1.7881233e-05, 4.3748852e-05, 9.655906e-06, 1.2755313e-05, 2.5748875e-05, 2.8967437e-05, 1.20400655e-05, 1.1086402e-05, 1.1444026e-05, 7.831743e-05, 1.060957e-05, 2.7656173e-05, 1.001353e-05, 1.0132739e-05, 1.0251946e-05, 1.0728779e-05, 1.6569955e-05, 1.668916e-05, 1.0728779e-05, 1.3232144e-05, 1.2516897e-05, 1.537788e-05, 1.1444026e-05, 1.0967195e-05, 1.0251946e-05, 1.490105e-05, 1.5139465e-05, 2.324554e-05, 0.0014568581, 1.7762026e-05, 1.1682442e-05, 1.2516897e-05, 1.2755313e-05, 64.29219, 1.001353e-05, 2.4556812e-05, 2.7298554e-05, 1.8715684e-05, 1.3112935e-05, 1.490105e-05, 1.6808368e-05, 0.00070249196, 1.001353e-05, 1.2278481e-05, 0.0013185388, 1.5258673e-05, 1.0847986e-05, 1.0967195e-05, 1.4305013e-05, 1.4066597e-05, 1.20400655e-05, 1.03711545e-05, 1.0967195e-05, 3.218599e-05, 1.2278481e-05, 8.2966224e-05, 2.1815062e-05, 2.3364748e-05, 1.1920858e-05, 2.2053475e-05, 1.3112935e-05, 1.239769e-05, 1.12056105e-05, 0.0009097593, 2.5868081e-05, 1.3232144e-05, 1.1563234e-05, 1.633154e-05, 1.4066597e-05, 1.20400655e-05, 1.3589766e-05, 1.2636105e-05, 2.7536968e-05, 1.1086402e-05, 3.933829e-05, 7.080781e-05, 1.3589766e-05, 1.6808368e-05, 2.527205e-05, 1.0490363e-05, 1.4305013e-05, 3.5285328e-05, 7.1165414e-05, 0.08424971, 1.060957e-05, 1.5258673e-05, 1.8238856e-05, 2.1695856e-05, 1.1920858e-05, 2.2411095e-05, 1.1920858e-05, 1.3589766e-05, 1.3828182e-05, 2.1219028e-05, 1.0132739e-05, 2.5868081e-05, 1.1682442e-05, 0.15507464, 0.0001904783, 1.3351351e-05, 5.2093103e-05, 1.7881233e-05, 1.6450747e-05, 5.8530048e-05, 1.3828182e-05, 0.007607891, 1.2636105e-05, 1.3351351e-05, 1.20400655e-05, 10.3583975, 1.3232144e-05, 1.1682442e-05, 1.03711545e-05, 1.060957e-05, 1.5973917e-05, 1.442422e-05, 1.0490363e-05, 1.1563234e-05, 0.00040768654, 1.3470559e-05, 0.0051801507, 1.5139465e-05, 5.2212305e-05, 1.3470559e-05, 1.2159274e-05, 2.8371409e-05, 9.570987, 1.2278481e-05, 1.4305013e-05, 3.7669426e-05, 1.1682442e-05, 1.668916e-05, 0.0009374991, 1.1444026e-05, 2.860982e-05, 74.458824, 5.2093103e-05, 0.0709143, 1.1086402e-05, 1.1920858e-05, 4.9470633e-05, 0.00014733183, 1.41858045e-05, 1.8954097e-05, 1.239769e-05, 1.2159274e-05, 1.001353e-05, 1.0847986e-05, 2.9921084e-05, 2.0622994e-05, 1.060957e-05, 1.2159274e-05, 1.2516897e-05, 4.804019e-05, 0.00050305587, 1.537788e-05, 1.20400655e-05, 1.28745205e-05, 1.5735503e-05, 1.490105e-05, 0.00051711506, 1.41858045e-05, 6.7112574e-05, 1.5973917e-05, 1.8358061e-05, 1.20400655e-05, 1.716599e-05, 1.4066597e-05, 1.180165e-05, 1.20400655e-05, 2.2530301e-05, 2.4556812e-05, 1.2755313e-05, 0.36738318, 9.7985234e-05, 1.20400655e-05, 0.05474068, 9.7751135e-06, 0.001404295, 5.5311582e-05, 1.12056105e-05, 2.0265374e-05, 1.442422e-05, 76.31356, 1.5020258e-05, 1.2755313e-05, 1.2516897e-05, 4.648912, 3.2543605e-05, 84.59731, 1.4781843e-05, 1.1563234e-05, 1.1324818e-05, 1.1682442e-05, 1.394739e-05, 9.417489e-06, 1.3589766e-05, 1.4662635e-05, 1.668916e-05, 1.3112935e-05, 1.6569955e-05, 1.6450747e-05, 2.5868081e-05, 1.8238856e-05, 87.03251, 0.0003591131, 1.5735503e-05, 1.2636105e-05, 1.001353e-05, 1.5735503e-05, 1.180165e-05, 5.8417315, 0.0012101313, 1.20400655e-05, 1.0847986e-05, 102.55488, 1.2159274e-05, 1.001353e-05, 0.010563847, 1.490105e-05, 1.0847986e-05, 8.4992615e-05, 1.1324818e-05, 1.2516897e-05, 1.239769e-05, 1.5735503e-05, 1.03711545e-05, 0.0022259583, 1.0967195e-05, 1.1444026e-05, 2.3483954e-05, 1.2516897e-05, 4.8278598e-05, 1.12056105e-05, 1.1682442e-05, 73.06266, 1.4781843e-05, 1.2516897e-05, 1.5258673e-05, 3.4570097e-05, 1.180165e-05, 9.184691, 1.5616295e-05, 1.585471e-05, 1.180165e-05, 1.3470559e-05, 1.180165e-05, 1.0728779e-05, 1.2516897e-05, 1.6450747e-05, 2.9563467e-05, 0.0007468057, 1.8954097e-05, 1.2516897e-05, 1.2755313e-05, 1.03711545e-05, 1.6093125e-05, 1.1563234e-05, 1.1444026e-05, 1.8358061e-05, 1.1682442e-05, 1.060957e-05, 1.28745205e-05, 1.180165e-05, 1.12056105e-05, 0.0009140504, 9.655906e-06, 1.20400655e-05, 1.12056105e-05, 1.12056105e-05, 1.668916e-05, 1.12056105e-05, 2.5033638e-05, 0.00059435784, 1.5497088e-05, 2.7536968e-05, 1.5616295e-05, 2.3126335e-05, 1.5616295e-05, 2.5629668e-05, 0.0002875282, 0.0054968493, 1.1682442e-05, 0.0001643761, 2.5033638e-05, 1.1086402e-05, 1.1444026e-05, 1.2636105e-05, 2.4437606e-05, 1.1324818e-05, 9.7751135e-06, 1.7404405e-05, 7.414543e-05, 1.239769e-05, 3.874227e-05, 0.0005267704, 83.31796, 1.4305013e-05, 3.623897e-05, 1.2993728e-05, 9.7751135e-06, 2.1219028e-05, 1.1324818e-05, 2.5748875e-05, 1.3351351e-05, 1.20400655e-05, 1.3232144e-05, 1.3589766e-05, 3.2066786e-05, 1.2636105e-05, 0.00056002784, 5.769563e-05, 1.1563234e-05, 1.7046783e-05, 0.06162845, 3.2782016e-05, 1.1444026e-05, 1.8596476e-05, 1.1920858e-05, 0.0016327808, 1.1086402e-05, 1.585471e-05, 1.1682442e-05, 1.1444026e-05, 83.33228, 0.005142912, 2.324554e-05, 1.2159274e-05, 2.1576649e-05, 1.1086402e-05, 0.00013767726, 4.2079995e-05, 1.03711545e-05, 1.966934e-05, 1.5020258e-05, 1.0132739e-05, 2.0384581e-05, 1.1324818e-05, 1.2636105e-05, 1.7642818e-05, 2.5033638e-05, 1.4781843e-05, 1.3351351e-05, 1.5139465e-05, 95.69973, 1.394739e-05, 1.4543428e-05, 3.8146245e-05, 0.00017414961, 2.0265374e-05, 54.830017, 1.5616295e-05, 5.7218822e-05, 1.2516897e-05, 1.1444026e-05, 1.4543428e-05, 1.28745205e-05, 1.060957e-05, 4.1448445, 1.1324818e-05, 1.3589766e-05, 0.007879026, 2.1695856e-05, 4.804019e-05, 1.1324818e-05, 2.1099822e-05, 1.0847986e-05, 4.5417706e-05, 1.0728779e-05, 1.0490363e-05, 1.180165e-05, 1.800044e-05, 1.1682442e-05, 1.7046783e-05, 4.8636208e-05, 1.2755313e-05, 2.2649509e-05, 0.00077694264, 0.0010771106, 9.655906e-06, 0.0017979734, 1.060957e-05, 1.28745205e-05, 1.2278481e-05, 2.0622994e-05, 101.31694, 53.983276, 0.00034760757, 0.0001753415, 1.5616295e-05, 1.0490363e-05, 1.0251946e-05, 2.0980615e-05, 1.1920858e-05, 1.3589766e-05, 1.3470559e-05, 1.3828182e-05, 3.1232346e-05, 1.41858045e-05, 2.7536968e-05, 3.8146245e-05, 1.41858045e-05, 1.5616295e-05, 6.12717e-05, 2.1576649e-05, 1.060957e-05, 2.1219028e-05, 1.28745205e-05, 8.201263e-05, 1.12056105e-05, 1.6808368e-05, 1.9907753e-05, 0.00027831495, 1.1682442e-05, 8.1774226e-05, 6.6333766, 0.00078102533, 8.60654e-05, 0.039228056, 1.1920858e-05, 1.0251946e-05, 1.490105e-05, 1.41858045e-05, 3.8980677e-05, 1.7881233e-05, 1.2159274e-05, 18.985126, 2.3364748e-05, 1.4066597e-05, 0.0024587135, 1.1324818e-05, 1.239769e-05, 9.4684305, 1.2755313e-05, 8.940657e-06, 5.578839e-05, 1.2636105e-05, 1.3470559e-05, 8.558861e-05, 1.180165e-05, 2.0384581e-05, 1.2516897e-05, 1.060957e-05, 0.0023782686, 1.2159274e-05, 1.1682442e-05, 0.0022313213, 0.0017679385, 1.1086402e-05, 1.2278481e-05, 1.1086402e-05, 2.9682673e-05, 1.28745205e-05, 1.3708975e-05, 1.2636105e-05, 1.2278481e-05, 0.00011109689, 1.28745205e-05, 1.7046783e-05, 1.4543428e-05, 1.2159274e-05, 0.0005607225, 22.110659, 1.6093125e-05, 1.1563234e-05, 1.28745205e-05, 1.3708975e-05, 1.3470559e-05, 1.2516897e-05, 2.324554e-05, 3.0636318e-05, 4.494089e-05, 1.060957e-05, 2.1219028e-05, 1.7881233e-05, 1.239769e-05, 1.3708975e-05, 1.8238856e-05, 2.0384581e-05, 6.305973e-05, 1.3232144e-05, 1.5020258e-05, 2.193427e-05, 1.1444026e-05, 1.28745205e-05, 1.12056105e-05, 1.2636105e-05, 2.2172682e-05, 1.5973917e-05, 0.0012669864, 0.004635809, 1.2755313e-05, 1.9073304e-05, 1.490105e-05, 2.5510462e-05, 1.4781843e-05, 8.928377e-05, 1.6093125e-05, 1.633154e-05, 0.0014907925, 1.4305013e-05, 1.4543428e-05, 1.1086402e-05, 3.5370965, 1.3828182e-05, 1.0728779e-05, 1.5973917e-05, 1.1324818e-05, 1.0967195e-05, 1.12056105e-05, 0.003088302, 1.5616295e-05, 1.180165e-05, 1.8358061e-05, 5.8410846e-05, 2.7298554e-05, 1.3232144e-05, 1.7881233e-05, 1.0728779e-05, 1.0490363e-05, 1.001353e-05, 13.308653, 0.0001315984, 1.060957e-05, 1.7404405e-05, 1.2278481e-05, 1.5020258e-05, 1.6212332e-05, 0.00048160876, 4.255681e-05, 9.7751135e-06, 1.585471e-05, 5.233151e-05, 97.21996, 98.86511, 1.7523613e-05, 0.00010096517, 0.59279346, 1.2755313e-05, 4.1722382e-05, 9.667406e-05, 7.872332, 1.8477269e-05, 1.28745205e-05, 1.1682442e-05, 1.1682442e-05, 1.5735503e-05, 1.1444026e-05, 1.9907753e-05, 1.0132739e-05, 1.9907753e-05, 1.8119648e-05, 1.3470559e-05, 6.1892967, 1.4305013e-05, 2.0622994e-05, 1.0251946e-05, 1.060957e-05, 1.1430157, 1.1563234e-05, 1.0967195e-05, 1.3112935e-05, 7.6648634e-05, 1.2993728e-05, 1.5616295e-05, 1.0728779e-05, 1.1682442e-05, 1.2159274e-05, 92.183334, 1.585471e-05, 5.2212305e-05, 1.9192512e-05, 2.6940936e-05, 3.886147e-05, 1.4543428e-05, 0.029040046, 2.2768714e-05, 1.5973917e-05, 72.69376, 1.5139465e-05, 2.1815062e-05, 3.623897e-05, 1.180165e-05, 1.2755313e-05, 1.0847986e-05, 1.239769e-05, 1.0132739e-05, 2.9086643e-05, 0.5755689, 1.0132739e-05, 1.2159274e-05, 1.1682442e-05, 1.3708975e-05, 1.537788e-05, 1.6808368e-05, 4.1722382e-05, 1.2636105e-05, 2.8252203e-05, 2.9086643e-05, 3.0755524e-05, 1.0847986e-05, 1.6569955e-05, 1.3232144e-05, 1.7285198e-05, 5.6026798e-05, 1.180165e-05, 5.0185852e-05, 5.9960475e-05, 1.060957e-05, 1.4066597e-05, 3.7907834e-05, 1.2278481e-05, 1.3470559e-05, 0.00010311072, 2.0384581e-05, 1.1563234e-05, 1.8119648e-05, 2.7298554e-05, 1.060957e-05, 69.88685, 1.4066597e-05, 1.442422e-05, 0.00021753329, 1.2993728e-05, 1.442422e-05, 1.6927575e-05, 1.2278481e-05, 1.180165e-05, 1.2516897e-05, 1.5616295e-05, 1.2636105e-05, 1.239769e-05, 0.00014292172, 1.7285198e-05, 1.3470559e-05, 1.0967195e-05, 1.12056105e-05, 1.001353e-05, 1.800044e-05, 3.70734e-05, 1.5139465e-05, 1.2516897e-05, 2.7179349e-05, 5.5430784e-05, 1.20400655e-05, 0.00033039355, 1.180165e-05, 2.7775379e-05, 3.6596583e-05, 6.413254e-05, 1.3351351e-05, 1.28745205e-05, 1.2159274e-05, 1.4781843e-05, 1.03711545e-05, 2.5391257e-05, 1.3828182e-05, 4.8993817e-05, 9.417489e-06, 1.239769e-05, 1.180165e-05, 1.2636105e-05, 1.41858045e-05, 1.537788e-05, 0.0068842066, 1.7881233e-05, 1.3470559e-05, 1.8119648e-05, 1.20400655e-05, 1.0728779e-05, 9.035656e-05, 12.850089, 1.3470559e-05, 1.0967195e-05, 1.1682442e-05, 79.61256, 1.0490363e-05, 1.8596476e-05, 1.3112935e-05, 1.0728779e-05, 2.741776e-05, 88.27163, 1.1563234e-05, 1.716599e-05, 1.5258673e-05, 0.00079818995, 1.4781843e-05, 0.00010334911, 3.778863e-05, 0.004808453, 1.5020258e-05, 8.574419, 1.4066597e-05, 5.0662664e-05, 9.512449e-05, 6.2225314e-05, 0.0010084541, 1.8358061e-05, 1.394739e-05, 1.6450747e-05, 1.20400655e-05, 1.1682442e-05, 1.3112935e-05, 9.894322e-06, 1.2159274e-05, 0.00024268066, 0.00010168035, 1.1920858e-05, 1.12056105e-05, 2.0384581e-05, 2.3722367e-05, 1.3828182e-05, 1.3112935e-05, 1.180165e-05, 1.3708975e-05, 1.1324818e-05, 1.1324818e-05, 0.00047634772, 2.193427e-05]

losses = np.sort(losses)
print(len(losses))
print(np.mean(losses))
print(np.mean(losses[:-32]))