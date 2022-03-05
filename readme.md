# Adaptive Auto Attack

"Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack"\
Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song\
CVPR 2022

**code and paper will be released soon**

A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free **Adaptive Auto Attack (A3)** evaluation method. We apply **A3** to over 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e, 1/10 on average (10x speed up), we achieve lower robust accuracy in all cases but one. Notably, we won **first place** out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method.

## News
+ [March 2022] The paper is accepted at CVPR 2022!

# Practical Adversarial Defenses Evaluation
A practical evaluation method should include several advantages:

* **Convenient (i.e., parameter-free)**
* **Efficient (i.e., fewer iterations)** 
* **Reliable (i.e., approaching the lower bound of robustness)**

Towards this target, we propose a parameter-free **Adaptive Auto Attack (A3)** evaluation method. 



## CIFAR-10 - Linf
The robust accuracy is evaluated at `eps = 8/255`, except for those marked with * for which `eps = 0.031`, where `eps` is the maximal Linf-norm allowed for the adversarial perturbations. The `eps` used is the same set in the original papers.

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**,10e6), while the “**<-**” column shows the iteration number of backward propagation(**million**,10e6). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

|#|paper|model |clean<br>(acc)|AA<br>(acc)|AA<br>(->)|AA<br>(<-)|A3<br>(acc)|A3<br>(->)|A3<br>(<-)|
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| WRN\ -70-16| 91.10| 65.88| 51.2 | 12.9 | **65.76(0.12)** | | |
|**2**| [(Rebuffi et al.,2021)](https://arxiv.org/abs/2103.01946) | WRN-70-16 | 88.54 | 64.25 | 50.8 | 12.6 |  | | |
|**3**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| WRN-28-10| 89.48| 62.80| 49.6 | 12.3 | **62.71(0.09)** | | |
|**4**| [(Rebuffi et al.,2021)](https://arxiv.org/abs/2103.01946)    | WRN-28-10 | 87.33 | 60.75 | 48.0 | 11.9 | **60.65(0.10)** | **4.14(11.59)** | **2.04(5.83)** |
|**5**| [(Sridhar et al.,2021)](https://arxiv.org/abs/2106.02078)‡ | WRN-34-15 | 86.53 | 60.41 | 47.5 | 11.8 | **60.31(0.10)** | **4.12(11.52)** | **2.02(5.84)** |
|**6**| [(Wu et al., 2020a)](https://arxiv.org/abs/2010.01279)‡| WRN-34-15| 87.67| 60.65|        |        |                 | | |
|**7**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)‡| WRN-28-10| 88.25| 60.04| 47.2 | 11.7 | **59.98(0.06)** | **4.09(11.54)** | **2.01(5.82)** |
|**8**| [(Sridhar et al.,2021)](https://arxiv.org/abs/2106.02078)‡ | WRN-28-10 | 89.46 | 59.66 | 47.1 | 11.7 | **59.51(0.15)** | **4.07(11.57)** | **2.00(5.85)** |
|**9**| [(Carmon et al., 2019)](https://arxiv.org/abs/1905.13736)‡| WRN-28-10| 89.69| 59.53| 47.1 | 11.7 | **59.43(0.10)** | **4.07(11.57)** | **2.00(5.85)** |
|**10**| [(Sehwag et al.,2021)](https://aisecure-workshop.github.io/aml-iclr2021/papers/39.pdf)‡ | WRN-34-10 | 85.85 | 59.09 | 46.7 | 11.6 | **58.99(0.10)** | **4.04(11.56)** | **1.98(5.86)** |
|**11**| [(Addepalli, et al.,2022)](https://openreview.net/pdf?id=eFP90pzlIz) | WRN-34-10 | 85.32 | 58.04 | 45.6 | 11.3 | **57.98(0.06)** | **3.95(11.54)** | **1.94(5.82)** |
|**12**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| WRN-70-16| 85.29| 57.20| 45.2 | 11.2 | | | |
|**13**| [(Sehwag et al., 2020)](https://github.com/fra31/auto-attack/issues/7)‡| WRN-28-10| 88.98| 57.14| 45.2 | 11.2 | **57.06(0.08)** | **3.91(11.56)** | **1.92(5.83)** |
|**14**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| WRN-34-20| 85.64| 56.86| 45.0 | 11.2 | **56.76(0.10)** | **3.88(11.60)** | **1.90(5.89)** |
|**15**| [(Wang et al., 2020)](https://openreview.net/forum?id=rklOg6EFwS)‡| WRN-28-10| 87.50| 56.29| 44.6 | 11.2 | **56.20(0.09)** | **3.86(11.55)** | **1.89(5.93)** |
|**16**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| WRN-34-10| 85.36| 56.17| | | | | |
|**17**| [(Alayrac et al., 2019)](https://arxiv.org/abs/1905.13725)‡| WRN-106-8| 86.46| 56.03| | | | | |
|**18**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| WRN-28-10| 87.11| 54.92| 43.4 | 10.8 | **54.76(0.16)** | **3.73(11.64)** | **1.83(5.90)** |
|19| [(Sehwag et al.,2021)](https://aisecure-workshop.github.io/aml-iclr2021/papers/39.pdf) | RN-18 | 84.38 | 54.43 | 43.2 | 10.7 | **54.31(0.12)** | **3.73(11.58)** | **1.83(23.61)** |
|**20**| [(Pang et al., 2020c)](https://arxiv.org/abs/2010.00467)| WRN-34-20| 86.43| 54.39| | | | | |
|**21**| [(Pang et al., 2020b)](https://arxiv.org/abs/2002.08619)| WRN-34-20| 85.14| 53.74| 43.0 | 10.7 | **53.67(0.07)** | **3.68(11.68)** | **1.81(5.91)** |
|**22**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| WRN-34-20| 88.70| 53.57| 43.0 | 10.7 | **53.45(0.12)** | **3.67(11.72)** | **1.80(5.94)** |
|**23**| [(Zhang et al., 2020b)](https://arxiv.org/abs/2002.11242)| WRN-34-10| 84.52| 53.51| 42.9 | 10.5 | **53.45(0.06)** | **3.66(11.72)** | **1.80(5.83)** |
|**24**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| WRN-34-20| 85.34| 53.42| 42.1 | 10.5 | **53.35(0.07)** | **3.66(11.50)** | **1.80(5.83)** |
|**25**| [(Huang et al., 2020)](https://arxiv.org/abs/2002.10319)\*| WRN-34-10| 83.48| 53.34| 42.1 | 10.5 | **53.19(0.15)** | **3.64(11.57)** | **1.79(5.87)** |
|**26**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)\*| WRN-34-10| 84.92| 53.08| 42.0 | 10.4 | **52.99(0.09)** | **3.63(11.57)** | **1.78(5.84)** |
|**27**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| WRN-34-10| 88.22| 52.86| 41.8 | 10.3 | **52.72(0.14)** | **3.62(11.55)** | **1.78(5.79)** |
|**28**| [(Qin et al., 2019)](https://arxiv.org/abs/1907.02610v2)| WRN-40-8| 86.28| 52.84| | | | | |
|**29**| [(Chen et al., 2020a)](https://arxiv.org/abs/2003.12862)| RN-50 (x3) | 86.04| 51.56| | | | | |
|**30**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| WRN-34-10| 85.32| 51.12| | | | | |
|**31**| [(Addepalli, et al.,2022)](https://openreview.net/pdf?id=eFP90pzlIz) | RN-18 | 80.24 | 51.06 | 40.5 | 10.2 | **51.02(0.04)** | **3.49(11.60)** | **1.71(5.96)** |
|**32**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/23)| WRN-34-10| 86.84| 50.72| 40.1 | 10.0 | **50.61(0.11)** | **3.48(11.53)** | **1.71(5.85)** |
|**33**| [(Engstrom et al., 2019)](https://github.com/MadryLab/robustness)| RN-50| 87.03| 49.25| 39.1 | 9.8 | **49.17(0.08)** | **3.40(11.49)** | **1.67(5.87)** |
|**34**| [(Kumari et al., 2019)](https://arxiv.org/abs/1905.05186)| WRN-34-10| 87.80| 49.12| | | | | |
|**35**| [(Mao et al., 2019)](http://papers.nips.cc/paper/8339-metric-learning-for-adversarial-robustness)| WRN-34-10| 86.21| 47.41| | | | | |
|**36**| [(Zhang et al., 2019a)](https://arxiv.org/abs/1905.00877)| WRN-34-10| 87.20| 44.83| 35.6 | 9.0 | **44.77(0.06)** | **3.09(11.52)** | **1.52(5.92)** |
|**37**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| WRN-34-10| 87.14| 44.04|  |  | | | |
|**38**| [(Pang et al., 2020a)](https://arxiv.org/abs/1905.10626)| RN-32| 80.89| 43.48| | | | | |
|**39**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| RN-18| 83.34| 43.21| | | | | |
|**40**| [(Shafahi et al., 2019)](https://arxiv.org/abs/1904.12843)| WRN-34-10| 86.11| 41.47| | | | | |
|**41**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| WRN-28-4| 84.36| 41.44| 33.3 | 8.6 | **40.73(0.71)** | **2.86(11.64)** | **1.40(6.14)** |
|**42**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)\*| RN-18| 81.30| 40.22| 32.7 | 8.7 | | | |
|**43**| [(Moosavi-Dezfooli et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Moosavi-Dezfooli_Robustness_via_Curvature_Regularization_and_Vice_Versa_CVPR_2019_paper)| WRN-28-10| 83.11| 38.50| | | | | |
|**44**| [(Zhang & Wang, 2019)](http://papers.nips.cc/paper/8459-defense-against-adversarial-attacks-using-feature-scattering-based-adversarial-training)| WRN-28-10| 89.98| 36.64| 30.0 | 8.2 | **36.31(0.33)** | **11.02(2.72)** | **5.44(1.51)** |
|**45**| [(Zhang & Xu, 2020)](https://openreview.net/forum?id=Syejj0NYvr&noteId=Syejj0NYvr)| WRN-28-10| 90.25| 36.45| 30.0 | 8.5 | **36.21(0.24)** | **11.21(2.68)** | **5.52(1.54)** |
|**46**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| RN-20| 78.91| 34.95| | | | | |
|**47**| [(Kim & Wang, 2020)](https://openreview.net/forum?id=rJlf_RVKwr)| WRN-34-10| 91.51| 34.22| 28.2 | 7.8 | **34.06(0.16)** | **10.47(2.69)** | **5.16(1.51)** |
|**48**| [(Wang & Zhang, 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Bilateral_Adversarial_Training_Towards_Fast_Training_of_More_Robust_Models_ICCV_2019_paper.html)| WRN-28-10| 92.80| 29.35| | | | | |
|**49**| [(Xiao et al., 2020)](https://arxiv.org/abs/1905.10510)\*| DenseNet-121| 79.28| 18.50| | | | | |
|**50**| [(Jin & Rinard, 2020)](https://arxiv.org/abs/2003.04286v1) | RN-18| 90.84| 1.35| 3.1 | 2.3 | **0.89(0.46)** | **2.24(1.38)** | **1.09(2.11)** |

## CIFAR-100 - Linf
The robust accuracy is computed at `eps = 8/255` in the Linf-norm, except for the models marked with * for which `eps = 0.031` is used. 

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The  “acc”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**), while the “**<-**” column shows the iteration number of backward propagation(**million**). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

|   #    | paper                                                        |   model   | clean<br>(acc) | AA<br>(acc) | AA<br>(->) | AA<br>(<-) |         A3(acc) |A3(->)  |A3(<-)  |
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
|**1**| [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)‡| WRN-70-16| 69.15| 36.88| 29.8 | 7.4 | **36.86(0.02)** | **2.54(11.73)** | **1.24(5.97)** |
|**2**| [(Rebuffi et al.,2021)](https://arxiv.org/abs/2103.01946) | WRN-70-16 | 63.56 | 34.64 | 28.0 | 7.0 | | | |
|**3**| [(Rebuffi et al.,2021)](https://arxiv.org/abs/2103.01946) | WRN-28-10 | 62.41 | 32.06 | 25.5 | 6.5 | **31.99(0.07)** | **2.24(11.38)** | **1.10(5.9)** |
|**4**| [(Addepalli, et al.,2022)](https://openreview.net/pdf?id=eFP90pzlIz) | WRN-34-10 | 65.73 | 30.35 | 24.3 | 6.1 | **30.31(0.04)** | **2.18(11.14)** | **1.07(5.7)** |
|**5**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| WRN-34-20| 62.55| 30.20| 24.0 | 6.1 | **30.12(0.08)** | **2.16(11.11)** | **1.05(5.8)** |
|**6**| [(Gowal et al. 2020)](https://arxiv.org/abs/2010.03593)| WRN-70-16| 60.86| 30.03| 23.93 | 6.09 | | | |
|**7**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| WRN-34-10| 60.64| 29.33| 23.21 | 5.94 | **29.16(0.17)** | **2.11(11.0)** | **1.03(5.77)** |
|**8**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| WRN-34-10| 60.38| 28.86| 23.01 | 5.84 | **28.78(0.08)** | **2.10(10.96)** | **1.02(5.72)** |
|**9**| [(Hendrycks et al., 2019)](https://arxiv.org/abs/1901.09960)‡| WRN-28-10| 59.23| 28.42| 22.74 | 5.73 | **28.29(0.13)** | **2.08(10.93)** | **1.02(5.61)** |
|**10**| [(Cui et al., 2020)](https://arxiv.org/abs/2011.11164)\*| WRN-34-10| 70.25| 27.16| | | | | |
|**11**| [(Chen et al., 2020b)](https://github.com/fra31/auto-attack/issues/26)| WRN-34-10| 62.15| 26.94| | | | | |
|**12**| [(Sitawarin et al., 2020)](https://github.com/fra31/auto-attack/issues/22)| WRN-34-10| 62.82| 24.57| 19.7 | 5.1 | **24.52(0.05)** | **1.90(10.36)** | **0.93(5.48)** |
|**13**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| RN-18| 53.83| 18.95| 15.3 | 4.0 | **18.87(0.08)** | **1.64(9.32)** | **0.80(5.0)** |

## MNIST - Linf
The robust accuracy is computed at `eps = 0.3` in the Linf-norm.

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**), while the “**<-**” column shows the iteration number of backward propagation(**million**). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

|   #    | paper                                                        | clean<br>(acc) | AA<br>(acc) |AA<br>(->)  |AA<br>(<-)  |A3<br>(acc)  |A3<br>(->)  |A3<br>(<-)  |
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| 99.26| 96.34|  76.05 | 18.44  | **96.31(0.03)** | **6.53(11.64)** | **3.22(5.72)** |
|**2**| [(Zhang et al., 2020a)](https://arxiv.org/abs/1906.06316)| 98.38| 93.96| | | | | |
|**3**| [(Gowal et al., 2019)](https://arxiv.org/abs/1810.12715)| 98.34| 92.83| | | | | |
|**4**| [(Zhang et al., 2019b)](https://arxiv.org/abs/1901.08573)| 99.48| 92.81|  73.12 | 17.88  | **92.71(0.05)** | **6.33(11.55)** | **3.12(5.73)** |
|**5**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| 98.95| 91.40| | | | | |
|**6**| [(Atzmon et al., 2019)](https://arxiv.org/abs/1905.11911)| 99.35| 90.85| | | | | |
|**7**| [(Madry et al., 2018)](https://arxiv.org/abs/1706.06083)| 98.53| 88.50| | | | | |
|**8**| [(Jang et al., 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.html)| 98.47| 87.99| | | | | |
|**9**| [(Wong et al., 2020)](https://arxiv.org/abs/2001.03994)| 98.50| 82.93| | | | | |
|**10**| [(Taghanaki et al., 2019)](http://openaccess.thecvf.com/content_CVPR_2019/html/Taghanaki_A_Kernelized_Manifold_Mapping_to_Diminish_the_Effect_of_Adversarial_CVPR_2019_paper.html)| 98.86| 0.00| | | | | |

## CIFAR-10 - L2
The robust accuracy is computed at `eps = 0.5` in the L2-norm.

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: ‡ indicates models which exploit additional data for training (e.g. unlabeled data, pre-training).

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**), while the “**<-**” column shows the iteration number of backward propagation(**million**). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

|   #    | paper                                                        |   model   | clean<br>(acc) | AA<br>(acc) |AA<br>(->)  |AA<br>(<-)  |A3<br>(acc)  |A3<br>(->)  |A3<br>(<-)  |
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
|**1**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)‡| WRN-70-16| 94.74| 80.53| | | | | |
|**2**| [(Rebuffi et al.,2021)](https://arxiv.org/abs/2103.01946) | WRN-28-10 | 91.79 | 78.80 | 62.00 | 15.20 | **78.79(0.01)** | **5.35(11.59)** | **2.63(5.78)** |
| **3**  | [(Sehwag et al.,2021)](https://aisecure-workshop.github.io/aml-iclr2021/papers/39.pdf) | WRN-34-10 | 90.31 | 76.11 | 59.89 | 14.69 | **76.10(0.01)** | **5.18(11.56)** | **2.55(5.76)** |
|**4**| [(Gowal et al., 2020)](https://arxiv.org/abs/2010.03593)| WRN-70-16| 90.90| 74.50| | | | | |
|**5**| [(Wu et al., 2020b)](https://arxiv.org/abs/2004.05884)| WRN-34-10| 88.51| 73.66| | | | | |
|**6**| [(Augustin et al., 2020)](https://arxiv.org/abs/2003.09461)‡| RN-50| 91.08| 72.91| | | | | |
|**7**| [(Engstrom et al., 2019)](https://github.com/MadryLab/robustness)| RN-50| 90.83| 69.24| 54.56 | 13.45 | **69.21(0.02)** | **4.72(11.56)** | **2.32(5.80)** |
|**8**| [(Rice et al., 2020)](https://arxiv.org/abs/2002.11569)| RN-18| 88.67| 67.68| 53.34 | 13.15 | **67.64(0.04)** | **4.61(11.57)** | **2.27(5.79)** |
|**9**| [(Rony et al., 2019)](https://arxiv.org/abs/1811.09600)| WRN-28-10| 89.05| 66.44| | | | | |
|**10**| [(Ding et al., 2020)](https://openreview.net/forum?id=HkeryxBtPB)| WRN-28-4| 88.02| 66.09| | | | | |

## ImageNet-Linf

The robust accuracy is computed at `eps = 4/255` in the Linf-norm.

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**), while the “**<-**” column shows the iteration number of backward propagation(**million**). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

| #     | paper                                                        | model | clean<br>(acc) | AA<br>(acc)   | AA<br>(->) | AA<br>(<-) | A3<br>(acc)         | A3<br>(->)         | A3<br>(<-)          |
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **1** | [(Wong et al.,2020)](https://arxiv.org/abs/2001.03994)       | RN-50        | 55.62 | **26.24** |        |        | 26.36           | 2.03           | 1.00            |
| **2** | [(Engstrom, et al.,2019)](https://github.com/MadryLab/robustness) | RN-50        | 62.50 | 29.18     | 11.66  | 2.98   | **29.14(0.04)** | **2.19(5.32)** | **1.08(2.76)**  |
| **3** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489)     | WRN-50-2     | 68.46 | 38.14     | 15.15  | 3.82   | **3812(0.02)**  | **2.67(5.67)** | **1.31(2.90)**  |
| **4** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489)     | RN-50        | 64.10 | 34.66     | 13.78  | 3.49   | **34.60(0.06)** | **2.47(5.58)** | **1.22(2.86)**  |
| **5** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489)     | RN-18        | 52.90 | 25.30     | 10.10  | 2.58   | **25.14(0.16)** | **1.96(5.15)** | **0.96(2.69)**  |
| **6** | Undefended                                                   | RN-50        | 76.74 | 0.0       | 0.40   | 0.39   | **0.0**         | **0.02(20.0)** | **0.005(78.0)** |

## ImageNet-L2

The robust accuracy is computed at `eps = 3.0` in the L2-norm.

**Note**: We will gradually refine the evaluation of the defense models.

**Note**: The  “**acc**”  column  shows  the  robust  accuracies  of  different  models. The “**->**” column shows the iteration number of forward propagation (**million**), while the “**<-**” column shows the iteration number of backward propagation(**million**). Notably, the “**acc**” column of **A3** shows the difference between the robust accuracies of **AA** and **A3**, the “**<-**” and “**->**” columns of **A3** show the speedup factors of **A3** relative to **AA**.

| #     | paper                                                    | model | clean<br>(acc) | AA<br>(acc) | AA<br>(->) | AA<br>(<-) | A3<br>(acc)         | A3<br>(->)         | A3<br>(<-)         |
|:---:|---|:---:|---:|---:|:---:|:---:|:---:|:---:|:---:|
| **1** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489) | DenseNet-161 | 66.14 | 36.52   | 14.51  | 3.67   | **36.50(0.02)** | **2.59(5.60)** | **1.28(2.87)** |
| **2** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489) | VGG16_BN     | 56.24 | 29.62   | 11.79  | 2.99   | **29.62(0.0)**  | **2.20(5.36)** | **1.08(2.77)** |
| **3** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489) | ShuffleNet   | 43.16 | 17.64   | 7.08   | 1.85   | **17.56(0.08)** | **1.58(4.48)** | **0.78(2.37)** |
| **4** | [(Salman et al.,2020)](https://arxiv.org/abs/2007.08489) | MobileNet-V2 | 49.62 | 24.78   | 9.89   | 2.52   | **24.74(0.04)** | **1.94(5.10)** | **0.95(2.65)** |

