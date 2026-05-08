Streaming from HuggingFaceFW/fineweb-edu / sample-100BT ...
num decayed parameter tensors: 51, with 162,915,840 parameters
num non-decayed parameter tensors: 75, with 125,521 parameters
using fused AdamW: True
Total parameters: 163.04M
Effective batch size: 32 (via 2 accumulation steps)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
README.md: 
 26.4k/? [00:00<00:00, 3.69MB/s]
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40231.61it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13304.99it/s]
Resolving data files: 100%
 2410/2410 [00:00<00:00, 21321.63it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16847.01it/s]
step 1: train 10.9294 | val 10.2394
step 100: train 7.5532 | val 7.5770
step 200: train 6.9536 | val 7.0204
step 300: train 6.7452 | val 6.7651
step 400: train 6.6058 | val 6.5353
step 500: train 6.3293 | val 6.5049
step 600: train 6.4780 | val 6.4731
step 700: train 6.1141 | val 6.2906
step 800: train 6.3104 | val 6.4736
step 900: train 5.9512 | val 6.1448
step 1000: train 6.0772 | val 6.0953
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 1100: train 5.9573 | val 5.9665
step 1200: train 5.8435 | val 5.8583
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37958.93it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17415.11it/s]
step 1300: train 5.9422 | val 5.8241
step 1400: train 5.5917 | val 5.7396
step 1500: train 5.8245 | val 5.6109
step 1600: train 5.6059 | val 5.6543
step 1700: train 5.5162 | val 5.5245
step 1800: train 5.5934 | val 5.3887
step 1900: train 5.5726 | val 5.4646
step 2000: train 5.3396 | val 5.3601
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 2100: train 5.3226 | val 5.3216
step 2200: train 5.2965 | val 5.4371
step 2300: train 5.1798 | val 5.1299
step 2400: train 5.0331 | val 5.1259
step 2500: train 5.0100 | val 5.1064
step 2600: train 4.8115 | val 5.0404
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38184.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8681.80it/s]
step 2700: train 5.3228 | val 4.9810
step 2800: train 4.9681 | val 5.0121
step 2900: train 4.8997 | val 4.9945
step 3000: train 4.7399 | val 4.9104
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 3100: train 4.7060 | val 4.8826
step 3200: train 4.8333 | val 4.8960
step 3300: train 4.6743 | val 4.8890
step 3400: train 4.7064 | val 4.6944
step 3500: train 4.6020 | val 4.8736
step 3600: train 4.7810 | val 4.6889
step 3700: train 4.4511 | val 4.6570
step 3800: train 4.6650 | val 4.5719
step 3900: train 4.4160 | val 4.5709
/usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:232: UserWarning: The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible. Please use `scheduler.step()` to step the scheduler. During the deprecation, if epoch is different from None, the closed form is used instead of the new chainable form, where available. Please open an issue if you are unable to replicate your use case: https://github.com/pytorch/pytorch/issues/new/choose.
  warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37395.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10769.02it/s]
step 4000: train 4.5863 | val 4.6069
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 4100: train 4.5168 | val 4.5756
step 4200: train 4.5748 | val 4.5115
step 4300: train 4.5283 | val 4.6619
step 4400: train 4.5863 | val 4.5075
step 4500: train 4.4856 | val 4.4506
step 4600: train 4.3389 | val 4.5249
step 4700: train 4.2500 | val 4.4678
step 4800: train 4.3638 | val 4.4116
step 4900: train 4.3389 | val 4.5420
step 5000: train 4.4668 | val 4.3178
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 5100: train 4.3976 | val 4.2906
step 5200: train 4.2873 | val 4.3606
step 5300: train 4.2611 | val 4.3378
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39993.64it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12816.82it/s]
step 5400: train 4.2426 | val 4.2764
step 5500: train 4.0685 | val 4.3202
step 5600: train 4.1750 | val 4.4108
step 5700: train 4.1250 | val 4.3117
step 5800: train 4.1601 | val 4.2959
step 5900: train 4.0863 | val 4.3285
step 6000: train 4.2023 | val 4.3835
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 6100: train 3.9620 | val 4.1920
step 6200: train 3.8234 | val 4.3610
step 6300: train 3.9635 | val 4.2424
step 6400: train 3.8907 | val 4.2262
step 6500: train 4.0788 | val 4.1590
step 6600: train 4.0020 | val 4.1586
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36904.44it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8445.31it/s]
step 6700: train 3.8815 | val 4.2149
step 6800: train 4.0719 | val 4.1801
step 6900: train 4.4095 | val 4.1732
step 7000: train 3.9639 | val 4.3743
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 7100: train 3.9327 | val 4.1805
step 7200: train 4.0371 | val 4.1591
step 7300: train 3.8275 | val 4.2107
step 7400: train 3.8796 | val 4.1949
step 7500: train 4.0456 | val 4.1152
step 7600: train 4.0404 | val 4.2612
step 7700: train 4.0326 | val 4.0228
step 7800: train 4.0365 | val 4.0383
step 7900: train 4.1274 | val 4.0677
step 8000: train 4.0482 | val 4.0876
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43182.61it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8940.22it/s]
step 8100: train 4.2739 | val 4.0174
step 8200: train 4.2257 | val 4.0648
step 8300: train 3.9717 | val 4.1664
step 8400: train 4.1758 | val 4.0591
step 8500: train 3.9384 | val 4.0629
step 8600: train 3.8919 | val 4.0698
step 8700: train 4.2093 | val 4.1374
step 8800: train 4.0429 | val 3.9366
step 8900: train 3.9538 | val 4.1027
step 9000: train 3.8323 | val 4.0015
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 9100: train 4.2984 | val 3.9771
step 9200: train 3.8029 | val 3.9119
step 9300: train 4.1889 | val 3.9493
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40460.77it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15455.14it/s]
step 9400: train 4.0622 | val 3.9794
step 9500: train 4.0165 | val 3.9392
step 9600: train 4.0078 | val 3.9579
step 9700: train 3.9410 | val 4.1426
step 9800: train 4.0265 | val 3.9976
step 9900: train 3.4819 | val 3.9641
step 10000: train 3.9023 | val 4.0106
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 10100: train 4.0096 | val 3.9983
step 10200: train 3.9913 | val 3.9417
step 10300: train 3.8528 | val 4.0789
step 10400: train 3.8341 | val 3.8631
step 10500: train 3.9554 | val 3.8824
step 10600: train 3.7790 | val 3.9246
step 10700: train 3.9474 | val 3.9501
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25536.00it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8626.96it/s]
step 10800: train 3.9221 | val 3.8819
step 10900: train 3.9099 | val 3.9241
step 11000: train 4.1314 | val 4.0383
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 11100: train 4.0098 | val 3.9571
step 11200: train 4.0379 | val 3.9559
step 11300: train 3.8687 | val 3.9802
step 11400: train 3.7142 | val 4.0534
step 11500: train 3.9981 | val 3.8571
step 11600: train 3.7138 | val 3.9779
step 11700: train 3.8963 | val 3.9162
step 11800: train 3.9570 | val 3.9186
step 11900: train 3.8264 | val 3.8461
step 12000: train 3.7819 | val 3.8559
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34645.37it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15952.69it/s]
step 12100: train 3.8382 | val 3.9300
step 12200: train 3.8331 | val 3.8922
step 12300: train 3.8456 | val 3.8771
step 12400: train 3.7726 | val 4.1023
step 12500: train 3.9522 | val 3.9120
step 12600: train 3.9560 | val 3.9042
step 12700: train 3.8485 | val 3.9424
step 12800: train 3.7509 | val 3.9520
step 12900: train 3.7368 | val 3.8402
step 13000: train 3.7560 | val 4.0032
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 13100: train 3.7716 | val 3.8019
step 13200: train 3.7060 | val 3.8186
step 13300: train 3.7296 | val 3.8710
step 13400: train 3.8059 | val 3.9241
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43642.75it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16487.51it/s]
step 13500: train 3.8229 | val 3.8319
step 13600: train 3.8710 | val 3.9111
step 13700: train 3.8136 | val 4.0109
step 13800: train 3.6298 | val 3.9301
step 13900: train 3.4900 | val 3.9144
step 14000: train 3.8280 | val 3.9524
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 14100: train 3.6595 | val 4.0179
step 14200: train 3.6449 | val 3.8008
step 14300: train 3.8442 | val 3.9411
step 14400: train 3.6472 | val 3.8794
step 14500: train 3.8561 | val 3.8633
step 14600: train 3.6380 | val 3.8052
step 14700: train 3.5016 | val 3.8156
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27310.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8288.55it/s]
step 14800: train 3.3626 | val 3.8913
step 14900: train 3.7759 | val 3.8587
step 15000: train 3.7956 | val 3.8480
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 15100: train 3.9298 | val 4.0671
step 15200: train 3.8518 | val 3.8565
step 15300: train 3.9067 | val 3.8354
step 15400: train 3.8760 | val 3.8757
step 15500: train 3.8709 | val 3.8704
step 15600: train 3.7814 | val 3.7554
step 15700: train 3.6561 | val 3.9384
step 15800: train 3.7548 | val 3.7294
step 15900: train 3.7196 | val 3.7456
step 16000: train 3.8504 | val 3.7820
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 16100: train 3.7922 | val 3.8164
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26988.50it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8536.78it/s]
step 16200: train 3.7542 | val 3.7305
step 16300: train 3.7597 | val 3.7878
step 16400: train 4.0308 | val 3.9047
step 16500: train 3.9543 | val 3.8173
step 16600: train 3.8230 | val 3.8031
step 16700: train 3.6824 | val 3.8606
step 16800: train 3.7572 | val 3.9072
step 16900: train 3.9009 | val 3.6853
step 17000: train 3.8520 | val 3.8493
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 17100: train 3.8287 | val 3.7812
step 17200: train 4.1234 | val 3.7780
step 17300: train 3.7335 | val 3.7146
step 17400: train 3.7789 | val 3.7200
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40112.99it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9208.71it/s]
step 17500: train 3.8010 | val 3.8053
step 17600: train 3.5681 | val 3.7442
step 17700: train 3.8605 | val 3.7553
step 17800: train 3.6245 | val 3.9656
step 17900: train 3.8134 | val 3.7922
step 18000: train 3.7769 | val 3.7836
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 18100: train 3.7360 | val 3.8284
step 18200: train 3.7919 | val 3.8065
step 18300: train 3.7025 | val 3.7119
step 18400: train 3.6022 | val 3.8974
step 18500: train 3.7521 | val 3.6870
step 18600: train 3.7274 | val 3.7073
step 18700: train 4.1101 | val 3.7575
step 18800: train 3.7093 | val 3.7849
Resolving data files: 100%
 2410/2410 [00:00<00:00, 46610.90it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10495.88it/s]
step 18900: train 3.6405 | val 3.7121
step 19000: train 3.8404 | val 3.7718
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 19100: train 3.7223 | val 3.8930
step 19200: train 3.6829 | val 3.7898
step 19300: train 3.7850 | val 3.7980
step 19400: train 3.7125 | val 3.8264
step 19500: train 3.7897 | val 3.9111
step 19600: train 3.5345 | val 3.6806
step 19700: train 3.6239 | val 3.8026
step 19800: train 3.5575 | val 3.7652
step 19900: train 3.5889 | val 3.7479
step 20000: train 3.8296 | val 3.6968
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 20100: train 3.7313 | val 3.7011
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45890.58it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8471.14it/s]
step 20200: train 3.7110 | val 3.7869
step 20300: train 3.6955 | val 3.7334
step 20400: train 3.7918 | val 3.7306
step 20500: train 3.6459 | val 3.9628
step 20600: train 3.6207 | val 3.7763
step 20700: train 3.4894 | val 3.7694
step 20800: train 3.5942 | val 3.8164
step 20900: train 3.5079 | val 3.8131
step 21000: train 3.7083 | val 3.7159
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 21100: train 3.5488 | val 3.8767
step 21200: train 3.7097 | val 3.6856
step 21300: train 3.6213 | val 3.7092
step 21400: train 3.7172 | val 3.7494
step 21500: train 3.7225 | val 3.7958
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43695.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10621.95it/s]
step 21600: train 3.5486 | val 3.7319
step 21700: train 3.7257 | val 3.7729
step 21800: train 3.8344 | val 3.9043
step 21900: train 3.3678 | val 3.8028
step 22000: train 3.4416 | val 3.7999
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 22100: train 3.6280 | val 3.8213
step 22200: train 3.4780 | val 3.8926
step 22300: train 3.6559 | val 3.6854
step 22400: train 3.5443 | val 3.8102
step 22500: train 3.8948 | val 3.7508
step 22600: train 3.6221 | val 3.7198
step 22700: train 3.8377 | val 3.6764
step 22800: train 3.7640 | val 3.6795
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28621.63it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16951.58it/s]
step 22900: train 3.6320 | val 3.7230
step 23000: train 3.6376 | val 3.6744
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 23100: train 3.9403 | val 3.6884
step 23200: train 3.8090 | val 3.8967
step 23300: train 3.7191 | val 3.7200
step 23400: train 3.6165 | val 3.7112
step 23500: train 3.7773 | val 3.7472
step 23600: train 3.8960 | val 3.7454
step 23700: train 3.7980 | val 3.6382
step 23800: train 3.8545 | val 3.8298
step 23900: train 3.7781 | val 3.6030
step 24000: train 3.6317 | val 3.6218
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 24100: train 3.3824 | val 3.6763
step 24200: train 3.5974 | val 3.6969
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39559.00it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8780.07it/s]
step 24300: train 3.7065 | val 3.6200
step 24400: train 3.5965 | val 3.6815
step 24500: train 3.7375 | val 3.8111
step 24600: train 3.7751 | val 3.7110
step 24700: train 3.7175 | val 3.7205
step 24800: train 3.5868 | val 3.7406
step 24900: train 3.6998 | val 3.8101
step 25000: train 3.7276 | val 3.5824
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 25100: train 3.5761 | val 3.7257
step 25200: train 3.6628 | val 3.6882
step 25300: train 3.8815 | val 3.6627
step 25400: train 3.4910 | val 3.6293
step 25500: train 3.6991 | val 3.6289
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26657.89it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 6917.14it/s]
step 25600: train 3.8080 | val 3.7019
step 25700: train 3.6450 | val 3.6530
step 25800: train 3.7132 | val 3.6537
step 25900: train 3.6557 | val 3.8737
step 26000: train 3.6943 | val 3.6966
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 26100: train 3.6360 | val 3.6888
step 26200: train 3.7518 | val 3.7397
step 26300: train 3.7439 | val 3.7213
step 26400: train 3.4983 | val 3.6134
step 26500: train 3.4953 | val 3.7906
step 26600: train 3.7669 | val 3.5885
step 26700: train 3.9206 | val 3.6281
step 26800: train 3.7412 | val 3.6743
step 26900: train 3.6925 | val 3.7064
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38874.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19816.50it/s]
step 27000: train 3.7001 | val 3.6359
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 27100: train 3.6345 | val 3.6899
step 27200: train 3.6151 | val 3.8127
step 27300: train 3.9663 | val 3.7145
step 27400: train 3.6663 | val 3.7221
step 27500: train 3.6271 | val 3.7514
step 27600: train 3.5867 | val 3.8355
step 27700: train 3.6457 | val 3.5980
step 27800: train 3.6947 | val 3.7457
step 27900: train 3.4577 | val 3.6800
step 28000: train 3.5982 | val 3.6772
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 28100: train 3.6429 | val 3.6331
step 28200: train 3.5697 | val 3.6350
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26361.25it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13391.78it/s]
step 28300: train 3.5377 | val 3.7179
step 28400: train 3.7792 | val 3.6582
step 28500: train 3.6460 | val 3.6637
step 28600: train 3.8145 | val 3.8884
step 28700: train 3.6606 | val 3.6962
step 28800: train 3.3337 | val 3.6998
step 28900: train 3.4114 | val 3.7658
step 29000: train 3.4878 | val 3.7560
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 29100: train 3.4857 | val 3.6572
step 29200: train 3.5274 | val 3.8321
step 29300: train 3.4460 | val 3.6095
step 29400: train 3.5200 | val 3.6484
step 29500: train 3.5789 | val 3.6892
step 29600: train 3.6594 | val 3.7302
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25879.63it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16123.97it/s]
step 29700: train 3.4393 | val 3.6504
step 29800: train 3.4441 | val 3.7081
step 29900: train 3.6503 | val 3.8275
step 30000: train 3.4789 | val 3.7365
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 30100: train 3.6179 | val 3.7378
step 30200: train 3.3894 | val 3.7622
step 30300: train 3.5126 | val 3.8425
step 30400: train 3.4948 | val 3.6168
step 30500: train 3.6250 | val 3.7600
step 30600: train 3.7255 | val 3.6871
step 30700: train 4.0423 | val 3.6638
step 30800: train 3.7383 | val 3.6009
step 30900: train 3.7668 | val 3.6149
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43946.34it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8706.26it/s]
step 31000: train 3.6245 | val 3.6681
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 31100: train 3.7295 | val 3.6268
step 31200: train 3.7842 | val 3.6286
step 31300: train 3.5523 | val 3.8385
step 31400: train 3.5556 | val 3.6636
step 31500: train 3.6816 | val 3.6634
step 31600: train 3.6266 | val 3.7021
step 31700: train 3.6560 | val 3.6856
step 31800: train 3.4652 | val 3.5755
step 31900: train 3.7095 | val 3.7653
step 32000: train 3.7384 | val 3.5471
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 32100: train 3.7438 | val 3.5861
step 32200: train 3.6789 | val 3.6192
step 32300: train 3.6580 | val 3.6512
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32001.88it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8680.39it/s]
step 32400: train 3.4917 | val 3.5819
step 32500: train 3.7385 | val 3.6352
step 32600: train 3.6367 | val 3.7614
step 32700: train 3.7520 | val 3.6718
step 32800: train 3.6197 | val 3.6837
step 32900: train 3.7713 | val 3.7107
step 33000: train 3.6522 | val 3.7806
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 33100: train 3.5510 | val 3.5537
step 33200: train 3.3031 | val 3.7048
step 33300: train 3.8007 | val 3.6476
step 33400: train 3.6381 | val 3.6183
step 33500: train 3.6910 | val 3.5860
step 33600: train 3.7434 | val 3.6007
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39558.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16191.99it/s]
step 33700: train 3.6261 | val 3.6543
step 33800: train 3.6861 | val 3.6147
step 33900: train 3.7200 | val 3.6198
step 34000: train 3.6304 | val 3.8350
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 34100: train 3.6043 | val 3.6571
step 34200: train 3.6139 | val 3.6504
step 34300: train 3.6059 | val 3.7113
step 34400: train 3.6306 | val 3.6945
step 34500: train 3.5775 | val 3.5987
step 34600: train 3.6421 | val 3.7778
step 34700: train 3.4680 | val 3.5611
step 34800: train 3.7100 | val 3.5932
step 34900: train 3.5579 | val 3.6322
step 35000: train 3.5962 | val 3.6786
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25433.39it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8567.05it/s]
step 35100: train 3.6532 | val 3.6067
step 35200: train 3.4615 | val 3.6579
step 35300: train 3.5705 | val 3.7804
step 35400: train 3.6300 | val 3.6879
step 35500: train 3.6154 | val 3.6879
step 35600: train 3.4369 | val 3.7105
step 35700: train 3.4733 | val 3.7930
step 35800: train 3.5798 | val 3.5669
step 35900: train 3.5969 | val 3.7201
step 36000: train 3.5456 | val 3.6556
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 36100: train 3.5704 | val 3.6390
step 36200: train 3.0984 | val 3.5995
step 36300: train 3.5588 | val 3.6088
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43526.61it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15392.34it/s]
step 36400: train 3.4734 | val 3.6849
step 36500: train 3.6660 | val 3.6350
step 36600: train 3.6387 | val 3.6513
step 36700: train 3.5454 | val 3.8825
step 36800: train 3.5443 | val 3.6860
step 36900: train 3.4636 | val 3.6668
step 37000: train 3.5550 | val 3.7327
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 37100: train 3.5682 | val 3.7167
step 37200: train 3.9269 | val 3.6168
step 37300: train 3.4474 | val 3.7762
step 37400: train 3.5416 | val 3.5806
step 37500: train 3.5486 | val 3.6151
step 37600: train 3.4450 | val 3.6527
step 37700: train 3.8761 | val 3.6958
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26149.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9212.03it/s]
step 37800: train 3.3479 | val 3.6101
step 37900: train 3.6284 | val 3.6851
step 38000: train 3.6354 | val 3.7868
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 38100: train 3.5419 | val 3.6751
step 38200: train 3.5844 | val 3.6825
step 38300: train 3.6961 | val 3.6863
step 38400: train 3.6772 | val 3.7527
step 38500: train 3.7817 | val 3.5292
step 38600: train 3.5761 | val 3.6736
step 38700: train 3.5939 | val 3.6213
step 38800: train 3.5805 | val 3.6025
step 38900: train 3.7079 | val 3.5695
step 39000: train 3.5252 | val 3.5630
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36450.90it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14816.00it/s]
step 39100: train 3.5249 | val 3.6297
step 39200: train 3.6748 | val 3.5857
step 39300: train 3.5359 | val 3.5950
step 39400: train 3.5875 | val 3.8068
step 39500: train 3.5401 | val 3.6313
step 39600: train 3.5364 | val 3.6253
step 39700: train 3.7396 | val 3.6632
step 39800: train 3.5275 | val 3.6432
step 39900: train 3.5768 | val 3.5559
step 40000: train 3.8995 | val 3.7267
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 40100: train 3.6349 | val 3.5256
step 40200: train 3.7306 | val 3.5573
step 40300: train 3.6254 | val 3.5990
step 40400: train 3.5752 | val 3.6372
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31939.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8648.69it/s]
step 40500: train 3.5174 | val 3.5581
step 40600: train 3.5558 | val 3.6190
step 40700: train 3.6349 | val 3.7309
step 40800: train 3.6358 | val 3.6487
step 40900: train 3.6758 | val 3.6510
step 41000: train 3.4823 | val 3.6826
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 41100: train 3.6662 | val 3.7449
step 41200: train 3.5299 | val 3.5260
step 41300: train 3.6543 | val 3.6771
step 41400: train 3.7843 | val 3.6155
step 41500: train 3.5980 | val 3.5979
step 41600: train 3.6178 | val 3.5706
step 41700: train 3.5499 | val 3.5668
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39516.78it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13871.04it/s]
step 41800: train 3.6823 | val 3.6383
step 41900: train 3.4828 | val 3.5853
step 42000: train 3.4901 | val 3.6071
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 42100: train 3.6185 | val 3.8070
step 42200: train 3.7644 | val 3.6318
step 42300: train 3.8048 | val 3.6281
step 42400: train 3.7565 | val 3.6809
step 42500: train 3.5314 | val 3.6645
step 42600: train 3.6131 | val 3.5611
step 42700: train 3.4428 | val 3.7327
step 42800: train 3.4808 | val 3.5328
step 42900: train 3.8171 | val 3.5772
step 43000: train 3.5872 | val 3.6132
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 43100: train 3.6206 | val 3.6455
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26727.25it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8087.41it/s]
step 43200: train 3.6978 | val 3.5660
step 43300: train 3.8827 | val 3.6318
step 43400: train 3.8496 | val 3.7392
step 43500: train 3.5433 | val 3.6511
step 43600: train 3.4521 | val 3.6667
step 43700: train 3.2740 | val 3.7008
step 43800: train 3.6742 | val 3.7775
step 43900: train 3.6381 | val 3.5587
step 44000: train 3.2288 | val 3.7003
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 44100: train 3.6205 | val 3.6369
step 44200: train 3.4835 | val 3.6285
step 44300: train 3.4032 | val 3.5816
step 44400: train 3.3693 | val 3.5824
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44061.27it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9037.36it/s]
step 44500: train 3.6895 | val 3.6632
step 44600: train 3.7137 | val 3.6194
step 44700: train 3.4528 | val 3.6395
step 44800: train 3.4072 | val 3.8658
step 44900: train 3.5618 | val 3.6570
step 45000: train 3.5626 | val 3.6500
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 45100: train 3.5111 | val 3.7094
step 45200: train 3.3975 | val 3.6885
step 45300: train 3.3146 | val 3.6092
step 45400: train 3.4780 | val 3.7405
step 45500: train 3.6280 | val 3.5408
step 45600: train 3.5765 | val 3.5563
step 45700: train 3.6163 | val 3.6007
step 45800: train 3.6417 | val 3.6280
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25724.66it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8248.39it/s]
step 45900: train 3.8821 | val 3.5477
step 46000: train 3.4019 | val 3.5909
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 46100: train 3.7772 | val 3.7141
step 46200: train 3.5937 | val 3.6189
step 46300: train 3.5746 | val 3.6290
step 46400: train 3.6908 | val 3.6463
step 46500: train 3.6477 | val 3.7135
step 46600: train 3.4252 | val 3.4970
step 46700: train 3.5192 | val 3.6486
step 46800: train 3.6009 | val 3.5784
step 46900: train 3.7480 | val 3.5458
step 47000: train 3.7359 | val 3.5273
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 47100: train 3.5275 | val 3.5267
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43947.87it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8701.23it/s]
step 47200: train 3.7317 | val 3.5913
step 47300: train 3.6891 | val 3.5284
step 47400: train 3.5970 | val 3.5453
step 47500: train 3.5302 | val 3.7572
step 47600: train 3.4139 | val 3.5866
step 47700: train 3.9624 | val 3.5851
step 47800: train 3.6980 | val 3.6217
step 47900: train 3.5485 | val 3.6123
step 48000: train 3.4974 | val 3.5178
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 48100: train 3.5068 | val 3.7024
step 48200: train 3.7183 | val 3.4851
step 48300: train 3.4953 | val 3.5203
step 48400: train 3.5584 | val 3.5673
step 48500: train 3.7518 | val 3.6157
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38982.32it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10481.83it/s]
step 48600: train 3.5515 | val 3.5305
step 48700: train 3.3970 | val 3.5801
step 48800: train 3.3655 | val 3.7059
step 48900: train 3.4421 | val 3.6163
step 49000: train 3.3746 | val 3.6282
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 49100: train 3.5543 | val 3.6476
step 49200: train 3.3371 | val 3.7158
step 49300: train 3.2995 | val 3.5022
step 49400: train 3.7333 | val 3.6421
step 49500: train 3.6360 | val 3.5857
step 49600: train 3.6073 | val 3.5696
step 49700: train 3.5939 | val 3.5352
step 49800: train 3.5945 | val 3.5364
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30571.10it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20760.94it/s]
step 49900: train 3.7022 | val 3.6159
step 50000: train 4.0617 | val 3.5674
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 50100: train 3.7855 | val 3.5806
step 50200: train 3.6397 | val 3.7988
step 50300: train 3.5703 | val 3.6159
step 50400: train 3.7613 | val 3.6014
step 50500: train 3.5847 | val 3.6539
step 50600: train 3.5408 | val 3.6366
step 50700: train 3.6042 | val 3.5486
step 50800: train 3.5692 | val 3.7066
step 50900: train 3.4626 | val 3.5049
step 51000: train 3.7109 | val 3.5477
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 51100: train 3.6545 | val 3.5848
step 51200: train 3.4248 | val 3.6298
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41712.67it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18759.27it/s]
step 51300: train 3.6580 | val 3.5447
step 51400: train 3.5992 | val 3.5997
step 51500: train 3.4851 | val 3.7281
step 51600: train 3.3030 | val 3.6329
step 51700: train 3.6828 | val 3.6465
step 51800: train 3.3750 | val 3.6790
step 51900: train 3.3030 | val 3.7511
step 52000: train 3.3720 | val 3.5223
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 52100: train 3.2492 | val 3.6830
step 52200: train 3.3807 | val 3.5995
step 52300: train 3.4280 | val 3.6063
step 52400: train 3.4295 | val 3.5657
step 52500: train 3.5737 | val 3.5679
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28438.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20413.08it/s]
step 52600: train 3.3953 | val 3.6310
step 52700: train 3.6310 | val 3.6122
step 52800: train 3.3652 | val 3.6018
step 52900: train 3.3652 | val 3.8336
step 53000: train 3.4234 | val 3.6351
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 53100: train 3.8182 | val 3.6267
step 53200: train 3.3440 | val 3.6742
step 53300: train 3.7587 | val 3.6629
step 53400: train 3.4205 | val 3.5630
step 53500: train 3.8553 | val 3.7260
step 53600: train 3.4837 | val 3.4974
step 53700: train 3.5617 | val 3.5376
step 53800: train 3.5611 | val 3.5665
step 53900: train 3.6307 | val 3.6157
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44427.87it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8644.36it/s]
step 54000: train 3.9667 | val 3.5181
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 54100: train 3.4424 | val 3.5647
step 54200: train 3.4645 | val 3.7080
step 54300: train 3.6516 | val 3.6105
step 54400: train 3.5579 | val 3.6045
step 54500: train 3.7117 | val 3.6280
step 54600: train 3.6414 | val 3.6977
step 54700: train 3.6307 | val 3.4763
step 54800: train 3.7581 | val 3.6507
step 54900: train 3.6373 | val 3.5624
step 55000: train 3.5327 | val 3.5254
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 55100: train 3.5569 | val 3.5067
step 55200: train 3.6987 | val 3.5135
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44691.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8747.64it/s]
step 55300: train 3.3998 | val 3.5728
step 55400: train 3.5988 | val 3.5231
step 55500: train 3.7159 | val 3.5480
step 55600: train 3.4969 | val 3.7788
step 55700: train 3.5743 | val 3.5904
step 55800: train 3.6399 | val 3.5772
step 55900: train 3.1765 | val 3.6416
step 56000: train 3.4128 | val 3.6115
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 56100: train 3.5152 | val 3.5129
step 56200: train 3.5661 | val 3.6921
step 56300: train 3.6597 | val 3.4782
step 56400: train 3.7621 | val 3.5242
step 56500: train 3.5430 | val 3.5535
step 56600: train 3.3937 | val 3.5941
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42473.52it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8472.00it/s]
step 56700: train 3.7224 | val 3.5215
step 56800: train 3.7953 | val 3.5754
step 56900: train 3.3504 | val 3.7056
step 57000: train 3.4731 | val 3.6039
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 57100: train 3.5710 | val 3.6075
step 57200: train 3.5144 | val 3.6408
step 57300: train 3.6205 | val 3.7149
step 57400: train 3.6465 | val 3.5166
step 57500: train 3.5692 | val 3.6301
step 57600: train 3.4339 | val 3.5767
step 57700: train 3.4918 | val 3.5673
step 57800: train 3.4363 | val 3.5326
step 57900: train 3.5295 | val 3.5279
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43687.83it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15412.55it/s]
step 58000: train 3.6401 | val 3.6062
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 58100: train 3.4848 | val 3.5575
step 58200: train 3.5548 | val 3.5721
step 58300: train 3.5681 | val 3.7896
step 58400: train 3.3985 | val 3.5968
step 58500: train 3.4948 | val 3.5917
step 58600: train 3.5333 | val 3.6473
step 58700: train 3.3449 | val 3.6308
step 58800: train 3.6025 | val 3.5285
step 58900: train 3.6386 | val 3.6945
step 59000: train 3.6331 | val 3.4952
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 59100: train 3.5012 | val 3.5402
step 59200: train 3.6309 | val 3.5773
step 59300: train 3.3540 | val 3.6328
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35684.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14856.86it/s]
step 59400: train 3.5040 | val 3.5339
step 59500: train 3.3533 | val 3.6056
step 59600: train 3.4120 | val 3.7391
step 59700: train 3.4996 | val 3.6329
step 59800: train 3.2938 | val 3.6499
step 59900: train 3.3859 | val 3.6643
step 60000: train 3.4107 | val 3.7369
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 60100: train 3.4535 | val 3.5194
step 60200: train 3.4811 | val 3.6742
step 60300: train 3.5698 | val 3.5993
step 60400: train 3.3503 | val 3.5870
step 60500: train 3.5053 | val 3.5510
step 60600: train 3.6303 | val 3.5515
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30325.30it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10355.94it/s]
step 60700: train 3.4779 | val 3.6208
step 60800: train 3.4880 | val 3.5792
step 60900: train 3.5499 | val 3.5976
step 61000: train 3.6484 | val 3.7946
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 61100: train 3.4309 | val 3.5853
step 61200: train 3.5807 | val 3.5686
step 61300: train 3.4612 | val 3.6268
step 61400: train 3.5820 | val 3.6005
step 61500: train 3.2802 | val 3.4898
step 61600: train 3.5489 | val 3.6804
step 61700: train 3.6257 | val 3.4699
step 61800: train 3.5424 | val 3.5002
step 61900: train 3.5021 | val 3.5400
step 62000: train 3.6895 | val 3.5803
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26107.15it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10362.15it/s]
step 62100: train 3.4679 | val 3.4925
step 62200: train 3.5181 | val 3.5462
step 62300: train 3.6093 | val 3.6788
step 62400: train 3.7158 | val 3.5778
step 62500: train 3.4831 | val 3.5873
step 62600: train 3.6755 | val 3.6043
step 62700: train 3.6826 | val 3.6708
step 62800: train 3.6683 | val 3.4514
step 62900: train 3.5692 | val 3.6067
step 63000: train 3.6352 | val 3.5515
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 63100: train 3.5967 | val 3.5358
step 63200: train 3.6280 | val 3.5016
step 63300: train 3.5175 | val 3.5001
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40954.03it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15731.31it/s]
step 63400: train 3.4660 | val 3.5693
step 63500: train 3.7185 | val 3.5189
step 63600: train 3.6223 | val 3.5366
step 63700: train 3.5450 | val 3.7579
step 63800: train 3.6476 | val 3.5625
step 63900: train 3.4633 | val 3.5562
step 64000: train 3.7661 | val 3.6251
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 64100: train 3.4898 | val 3.5788
step 64200: train 3.4130 | val 3.4936
step 64300: train 3.6428 | val 3.6622
step 64400: train 3.8247 | val 3.4628
step 64500: train 3.5241 | val 3.5028
step 64600: train 3.4611 | val 3.5490
step 64700: train 3.5803 | val 3.5829
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25729.31it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18747.29it/s]
step 64800: train 3.3730 | val 3.5032
step 64900: train 3.7234 | val 3.5648
step 65000: train 3.6942 | val 3.6913
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 65100: train 3.6605 | val 3.5962
step 65200: train 3.6375 | val 3.6048
step 65300: train 3.3952 | val 3.6312
step 65400: train 3.3457 | val 3.6931
step 65500: train 3.7066 | val 3.4770
step 65600: train 3.5975 | val 3.6125
step 65700: train 3.4943 | val 3.5598
step 65800: train 3.3593 | val 3.5488
step 65900: train 3.6940 | val 3.5039
step 66000: train 3.6545 | val 3.5103
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41352.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9651.43it/s]
step 66100: train 3.4091 | val 3.5822
step 66200: train 3.8235 | val 3.5276
step 66300: train 3.6167 | val 3.5536
step 66400: train 3.4126 | val 3.7774
step 66500: train 3.5832 | val 3.5912
step 66600: train 3.1208 | val 3.5851
step 66700: train 3.6893 | val 3.6398
step 66800: train 3.3157 | val 3.6226
step 66900: train 3.5003 | val 3.5357
step 67000: train 3.4529 | val 3.6906
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 67100: train 3.2754 | val 3.5016
step 67200: train 3.4515 | val 3.5366
step 67300: train 3.1968 | val 3.5735
step 67400: train 3.4572 | val 3.6058
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26535.77it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8521.05it/s]
step 67500: train 3.2583 | val 3.5250
step 67600: train 3.4099 | val 3.5966
step 67700: train 3.5114 | val 3.7327
step 67800: train 3.5723 | val 3.6222
step 67900: train 3.5731 | val 3.6290
step 68000: train 3.3035 | val 3.6620
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 68100: train 3.3735 | val 3.7230
step 68200: train 3.6129 | val 3.4962
step 68300: train 3.3028 | val 3.6452
step 68400: train 3.5199 | val 3.5537
step 68500: train 3.6157 | val 3.5428
step 68600: train 3.2777 | val 3.4979
step 68700: train 3.6571 | val 3.5056
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26831.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15134.48it/s]
step 68800: train 3.4576 | val 3.5604
step 68900: train 3.5543 | val 3.5142
step 69000: train 3.8291 | val 3.5185
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 69100: train 3.7053 | val 3.7332
step 69200: train 3.4043 | val 3.5547
step 69300: train 3.3913 | val 3.5453
step 69400: train 3.6188 | val 3.5992
step 69500: train 3.7083 | val 3.5730
step 69600: train 3.5990 | val 3.4620
step 69700: train 3.6810 | val 3.6454
step 69800: train 3.3953 | val 3.4380
step 69900: train 3.5740 | val 3.4695
step 70000: train 3.4307 | val 3.5067
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 70100: train 3.3609 | val 3.5544
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25144.08it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8727.74it/s]
step 70200: train 3.8690 | val 3.4627
step 70300: train 3.4748 | val 3.5264
step 70400: train 3.4286 | val 3.6741
step 70500: train 3.4362 | val 3.5485
step 70600: train 3.4291 | val 3.5717
step 70700: train 3.4497 | val 3.5857
step 70800: train 3.6151 | val 3.6586
step 70900: train 3.5738 | val 3.4266
step 71000: train 3.4954 | val 3.5892
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 71100: train 3.5549 | val 3.5250
step 71200: train 3.5492 | val 3.4973
step 71300: train 3.5226 | val 3.4845
step 71400: train 3.5425 | val 3.4973
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24619.32it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9030.69it/s]
step 71500: train 3.7127 | val 3.5457
step 71600: train 3.4501 | val 3.4947
step 71700: train 3.3670 | val 3.5189
step 71800: train 3.6910 | val 3.7457
step 71900: train 3.3836 | val 3.5580
step 72000: train 3.0748 | val 3.5445
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 72100: train 3.5364 | val 3.5960
step 72200: train 3.5878 | val 3.5760
step 72300: train 3.4828 | val 3.4781
step 72400: train 3.6172 | val 3.6638
step 72500: train 3.5377 | val 3.4425
step 72600: train 3.6324 | val 3.4917
step 72700: train 3.2933 | val 3.5311
step 72800: train 3.3831 | val 3.5664
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27978.74it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15582.28it/s]
step 72900: train 3.4167 | val 3.4880
step 73000: train 3.5092 | val 3.5479
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 73100: train 3.5899 | val 3.6759
step 73200: train 3.5191 | val 3.5774
step 73300: train 3.3676 | val 3.5911
step 73400: train 3.4777 | val 3.6158
step 73500: train 3.5687 | val 3.6849
step 73600: train 3.5217 | val 3.4636
step 73700: train 3.5659 | val 3.6211
step 73800: train 3.8155 | val 3.5504
step 73900: train 3.4648 | val 3.5329
step 74000: train 3.5043 | val 3.5123
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 74100: train 3.5171 | val 3.4952
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40326.79it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17167.66it/s]
step 74200: train 3.4197 | val 3.5666
step 74300: train 3.6073 | val 3.5192
step 74400: train 3.5261 | val 3.5457
step 74500: train 3.1147 | val 3.7651
step 74600: train 3.2259 | val 3.5724
step 74700: train 3.5655 | val 3.5738
step 74800: train 3.4318 | val 3.6353
step 74900: train 3.5085 | val 3.6145
step 75000: train 3.5359 | val 3.5203
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 75100: train 3.4598 | val 3.6842
step 75200: train 3.3516 | val 3.4740
step 75300: train 3.4116 | val 3.5197
step 75400: train 3.3784 | val 3.5509
step 75500: train 3.6539 | val 3.6103
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27882.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16503.73it/s]
step 75600: train 3.2835 | val 3.5098
step 75700: train 3.5677 | val 3.5816
step 75800: train 3.3951 | val 3.7216
step 75900: train 3.4305 | val 3.6145
step 76000: train 3.3677 | val 3.6146
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 76100: train 3.4139 | val 3.6402
step 76200: train 3.5369 | val 3.7222
step 76300: train 3.3162 | val 3.4930
step 76400: train 3.4452 | val 3.6189
step 76500: train 3.5724 | val 3.5565
step 76600: train 3.5934 | val 3.5371
step 76700: train 3.7597 | val 3.4893
step 76800: train 3.6361 | val 3.4859
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30386.47it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19519.41it/s]
step 76900: train 3.5213 | val 3.5637
step 77000: train 3.6226 | val 3.4981
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 77100: train 3.5889 | val 3.5208
step 77200: train 3.6319 | val 3.7255
step 77300: train 3.6115 | val 3.5514
step 77400: train 3.5339 | val 3.5383
step 77500: train 3.6212 | val 3.5855
step 77600: train 3.8888 | val 3.5544
step 77700: train 3.6798 | val 3.4600
step 77800: train 3.7555 | val 3.6485
step 77900: train 3.5142 | val 3.4229
step 78000: train 3.5806 | val 3.4596
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 78100: train 3.6251 | val 3.5093
step 78200: train 3.4679 | val 3.5468
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41410.71it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15721.62it/s]
step 78300: train 3.7082 | val 3.4574
step 78400: train 3.6499 | val 3.5186
step 78500: train 3.6280 | val 3.6503
step 78600: train 3.5818 | val 3.5643
step 78700: train 3.4684 | val 3.5707
step 78800: train 3.7535 | val 3.5827
step 78900: train 3.4108 | val 3.6509
step 79000: train 3.7536 | val 3.4323
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 79100: train 3.7775 | val 3.5911
step 79200: train 3.5741 | val 3.5285
step 79300: train 3.5900 | val 3.4961
step 79400: train 3.5773 | val 3.4752
step 79500: train 3.4759 | val 3.4802
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25500.57it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11103.60it/s]
step 79600: train 3.5068 | val 3.5528
step 79700: train 3.7150 | val 3.4856
step 79800: train 3.6030 | val 3.5077
step 79900: train 3.4210 | val 3.7255
step 80000: train 3.5475 | val 3.5416
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 80100: train 3.2702 | val 3.5356
step 80200: train 3.5771 | val 3.5840
step 80300: train 3.6583 | val 3.5503
step 80400: train 3.4609 | val 3.4778
step 80500: train 3.4881 | val 3.6468
step 80600: train 3.4906 | val 3.4435
step 80700: train 3.5742 | val 3.5003
step 80800: train 3.4639 | val 3.5228
step 80900: train 3.6145 | val 3.5695
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36204.29it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16490.75it/s]
step 81000: train 3.5415 | val 3.4876
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 81100: train 3.5987 | val 3.5487
step 81200: train 3.5235 | val 3.6735
step 81300: train 3.4838 | val 3.5781
step 81400: train 3.4632 | val 3.5835
step 81500: train 3.5902 | val 3.6108
step 81600: train 3.5642 | val 3.6832
step 81700: train 3.7749 | val 3.4582
step 81800: train 3.5338 | val 3.6092
step 81900: train 3.5900 | val 3.5409
step 82000: train 3.3915 | val 3.5300
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 82100: train 3.3886 | val 3.4945
step 82200: train 3.4104 | val 3.5008
Resolving data files: 100%
 2410/2410 [00:00<00:00, 46902.00it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8286.92it/s]
step 82300: train 3.3236 | val 3.5811
step 82400: train 3.4218 | val 3.5283
step 82500: train 3.5275 | val 3.5531
step 82600: train 3.7297 | val 3.7732
step 82700: train 3.3660 | val 3.5785
step 82800: train 3.4259 | val 3.5788
step 82900: train 3.3792 | val 3.6276
step 83000: train 3.3727 | val 3.6012
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 83100: train 3.5645 | val 3.5172
step 83200: train 3.6373 | val 3.6817
step 83300: train 3.4643 | val 3.4728
step 83400: train 3.2550 | val 3.5307
step 83500: train 3.5068 | val 3.5505
step 83600: train 3.3843 | val 3.5990
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40599.06it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11675.17it/s]
step 83700: train 3.5003 | val 3.5134
step 83800: train 3.4636 | val 3.5786
step 83900: train 3.4182 | val 3.7036
step 84000: train 3.4357 | val 3.6066
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 84100: train 3.4601 | val 3.5962
step 84200: train 3.7999 | val 3.5967
step 84300: train 3.6359 | val 3.6648
step 84400: train 3.5499 | val 3.4425
step 84500: train 3.4802 | val 3.5850
step 84600: train 3.3687 | val 3.5347
step 84700: train 3.8097 | val 3.5160
step 84800: train 3.4542 | val 3.4787
step 84900: train 3.5696 | val 3.4743
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27348.26it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12107.27it/s]
step 85000: train 3.5646 | val 3.5342
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 85100: train 3.6794 | val 3.4818
step 85200: train 3.5677 | val 3.4997
step 85300: train 3.5240 | val 3.7213
step 85400: train 3.4995 | val 3.5288
step 85500: train 3.5029 | val 3.5263
step 85600: train 3.5436 | val 3.5718
step 85700: train 3.5691 | val 3.5393
step 85800: train 3.6282 | val 3.4506
step 85900: train 3.6175 | val 3.6451
step 86000: train 3.4215 | val 3.4342
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 86100: train 3.6193 | val 3.4750
step 86200: train 3.5349 | val 3.5016
step 86300: train 3.4392 | val 3.5470
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32138.32it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13517.24it/s]
step 86400: train 3.4702 | val 3.4570
step 86500: train 3.6541 | val 3.5249
step 86600: train 3.5662 | val 3.6507
step 86700: train 3.4891 | val 3.5510
step 86800: train 3.6421 | val 3.5567
step 86900: train 3.5964 | val 3.5816
step 87000: train 3.5140 | val 3.6468
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 87100: train 3.4222 | val 3.4309
step 87200: train 3.4877 | val 3.5747
step 87300: train 3.6446 | val 3.5424
step 87400: train 3.6298 | val 3.5053
step 87500: train 3.5800 | val 3.4851
step 87600: train 3.5253 | val 3.4759
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32723.55it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19217.26it/s]
step 87700: train 3.6038 | val 3.5434
step 87800: train 3.8781 | val 3.4941
step 87900: train 3.4865 | val 3.5149
step 88000: train 3.5262 | val 3.7417
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 88100: train 3.5756 | val 3.5471
step 88200: train 3.7167 | val 3.5422
step 88300: train 3.5755 | val 3.5956
step 88400: train 3.4763 | val 3.5859
step 88500: train 3.3025 | val 3.4856
step 88600: train 3.2402 | val 3.6494
step 88700: train 3.6370 | val 3.4451
step 88800: train 3.4711 | val 3.4827
step 88900: train 3.4893 | val 3.5152
step 89000: train 3.7833 | val 3.5729
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26598.97it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15475.10it/s]
step 89100: train 3.6646 | val 3.4806
step 89200: train 3.5015 | val 3.5396
step 89300: train 3.5296 | val 3.6764
step 89400: train 3.5202 | val 3.5782
step 89500: train 3.5637 | val 3.5729
step 89600: train 3.3898 | val 3.6101
step 89700: train 3.6137 | val 3.6851
step 89800: train 3.5074 | val 3.4713
step 89900: train 3.4922 | val 3.6042
step 90000: train 3.2705 | val 3.5525
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 90100: train 3.2029 | val 3.5401
step 90200: train 3.4014 | val 3.5015
step 90300: train 3.2915 | val 3.4993
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34943.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13629.25it/s]
step 90400: train 3.3022 | val 3.5765
step 90500: train 3.7523 | val 3.5331
step 90600: train 3.5101 | val 3.5371
step 90700: train 3.4828 | val 3.7680
step 90800: train 3.3109 | val 3.5725
step 90900: train 3.3170 | val 3.5672
step 91000: train 3.5438 | val 3.6214
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 91100: train 3.3384 | val 3.5938
step 91200: train 3.4210 | val 3.5152
step 91300: train 3.3840 | val 3.6723
step 91400: train 3.7952 | val 3.4621
step 91500: train 3.7379 | val 3.4885
step 91600: train 3.5911 | val 3.5160
step 91700: train 3.5039 | val 3.5556
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25750.15it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8516.11it/s]
step 91800: train 3.5101 | val 3.4540
step 91900: train 3.4766 | val 3.5185
step 92000: train 3.3441 | val 3.6440
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 92100: train 3.4580 | val 3.5475
step 92200: train 3.5666 | val 3.5503
step 92300: train 3.7068 | val 3.5618
step 92400: train 3.5398 | val 3.6379
step 92500: train 3.5098 | val 3.4187
step 92600: train 3.4545 | val 3.5650
step 92700: train 3.3491 | val 3.5017
step 92800: train 3.3979 | val 3.4775
step 92900: train 3.5100 | val 3.4438
step 93000: train 3.5127 | val 3.4496
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40543.69it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9577.13it/s]
step 93100: train 3.5467 | val 3.5220
step 93200: train 3.4000 | val 3.4566
step 93300: train 3.5709 | val 3.4845
step 93400: train 3.3069 | val 3.7009
step 93500: train 3.4690 | val 3.5154
step 93600: train 3.4256 | val 3.5135
step 93700: train 3.6992 | val 3.5520
step 93800: train 3.4592 | val 3.5371
step 93900: train 3.1511 | val 3.4425
step 94000: train 3.4009 | val 3.6301
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 94100: train 3.4976 | val 3.4042
step 94200: train 3.3253 | val 3.4581
step 94300: train 3.5358 | val 3.4856
step 94400: train 3.5212 | val 3.5326
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44574.02it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10165.19it/s]
step 94500: train 3.7029 | val 3.4438
step 94600: train 3.7497 | val 3.5117
step 94700: train 3.5235 | val 3.6364
step 94800: train 3.5090 | val 3.5501
step 94900: train 3.6757 | val 3.5452
step 95000: train 3.5428 | val 3.5742
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 95100: train 3.5826 | val 3.6480
step 95200: train 3.7471 | val 3.4179
step 95300: train 3.5814 | val 3.5579
step 95400: train 3.4008 | val 3.5028
step 95500: train 3.7910 | val 3.4921
step 95600: train 3.6626 | val 3.4533
step 95700: train 3.4549 | val 3.4653
Resolving data files: 100%
 2410/2410 [00:00<00:00, 19594.93it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14998.41it/s]
step 95800: train 3.5184 | val 3.5406
step 95900: train 3.3781 | val 3.4870
step 96000: train 3.5423 | val 3.5093
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 96100: train 3.4706 | val 3.7305
step 96200: train 3.5036 | val 3.5457
step 96300: train 3.7830 | val 3.5358
step 96400: train 3.4959 | val 3.5955
step 96500: train 3.3958 | val 3.5324
step 96600: train 3.5112 | val 3.4719
step 96700: train 3.5929 | val 3.6365
step 96800: train 3.5454 | val 3.4250
step 96900: train 3.5680 | val 3.4768
step 97000: train 3.4920 | val 3.5056
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 97100: train 3.3681 | val 3.5677
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39123.09it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14876.43it/s]
step 97200: train 3.4727 | val 3.4729
step 97300: train 3.4364 | val 3.5374
step 97400: train 3.2532 | val 3.6661
step 97500: train 3.4399 | val 3.5659
step 97600: train 3.2740 | val 3.5709
step 97700: train 3.4491 | val 3.5956
step 97800: train 3.5502 | val 3.6726
step 97900: train 3.2991 | val 3.4460
step 98000: train 3.2355 | val 3.5977
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 98100: train 3.5495 | val 3.5324
step 98200: train 3.3950 | val 3.5359
step 98300: train 3.3337 | val 3.4917
step 98400: train 3.3148 | val 3.4949
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45514.51it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9255.74it/s]
step 98500: train 3.4450 | val 3.5712
step 98600: train 3.4664 | val 3.5167
step 98700: train 3.3897 | val 3.5356
step 98800: train 3.4391 | val 3.7689
step 98900: train 3.3070 | val 3.5566
step 99000: train 3.4442 | val 3.5611
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 99100: train 3.3777 | val 3.6069
step 99200: train 3.3825 | val 3.5792
step 99300: train 3.6897 | val 3.4979
step 99400: train 3.3093 | val 3.6628
step 99500: train 3.3869 | val 3.4283
step 99600: train 3.5521 | val 3.4611
step 99700: train 3.2909 | val 3.4925
step 99800: train 3.7681 | val 3.5389
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36786.38it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15648.72it/s]
step 99900: train 4.0829 | val 3.4582
step 100000: train 3.4020 | val 3.5088
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 100100: train 3.3908 | val 3.6352
step 100200: train 4.1935 | val 3.5286
step 100300: train 3.6940 | val 3.5367
step 100400: train 3.4456 | val 3.5568
step 100500: train 3.5889 | val 3.6247
step 100600: train 3.3681 | val 3.3841
step 100700: train 3.5637 | val 3.5377
step 100800: train 3.5933 | val 3.4923
step 100900: train 3.5573 | val 3.4714
step 101000: train 4.0003 | val 3.4348
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 101100: train 3.4983 | val 3.4447
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39929.97it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16104.51it/s]
step 101200: train 3.3796 | val 3.5038
step 101300: train 3.3616 | val 3.4551
step 101400: train 3.5128 | val 3.4674
step 101500: train 3.4179 | val 3.7075
step 101600: train 3.9014 | val 3.5169
step 101700: train 3.3833 | val 3.5120
step 101800: train 3.6029 | val 3.5224
step 101900: train 3.5453 | val 3.5261
step 102000: train 3.4801 | val 3.4335
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 102100: train 3.3936 | val 3.6319
step 102200: train 3.3036 | val 3.4097
step 102300: train 3.4293 | val 3.4526
step 102400: train 3.7773 | val 3.4917
step 102500: train 3.3236 | val 3.5223
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28637.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8286.21it/s]
step 102600: train 3.6119 | val 3.4483
step 102700: train 3.3941 | val 3.4971
step 102800: train 3.5631 | val 3.6177
step 102900: train 3.6146 | val 3.5290
step 103000: train 3.4408 | val 3.5414
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 103100: train 3.4363 | val 3.5509
step 103200: train 3.4119 | val 3.6331
step 103300: train 3.5728 | val 3.4059
step 103400: train 3.4844 | val 3.5654
step 103500: train 3.4821 | val 3.5011
step 103600: train 3.6089 | val 3.4938
step 103700: train 3.3658 | val 3.4616
step 103800: train 3.6538 | val 3.4601
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44886.75it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19256.33it/s]
step 103900: train 3.4641 | val 3.5212
step 104000: train 3.4098 | val 3.4746
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 104100: train 3.5751 | val 3.4985
step 104200: train 3.5122 | val 3.7215
step 104300: train 3.3556 | val 3.5261
step 104400: train 3.4467 | val 3.5266
step 104500: train 3.4998 | val 3.5642
step 104600: train 3.4944 | val 3.5323
step 104700: train 3.5347 | val 3.4794
step 104800: train 3.4669 | val 3.6218
step 104900: train 3.2889 | val 3.4203
step 105000: train 3.3927 | val 3.4590
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 105100: train 3.4392 | val 3.5226
step 105200: train 3.1709 | val 3.5585
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40101.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15497.97it/s]
step 105300: train 3.5101 | val 3.4707
step 105400: train 3.4482 | val 3.5369
step 105500: train 3.4629 | val 3.6634
step 105600: train 3.4679 | val 3.5613
step 105700: train 3.3676 | val 3.5711
step 105800: train 3.3783 | val 3.5889
step 105900: train 3.3506 | val 3.6690
step 106000: train 3.4552 | val 3.4369
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 106100: train 3.2583 | val 3.5907
step 106200: train 3.5516 | val 3.5255
step 106300: train 3.3158 | val 3.5324
step 106400: train 3.4602 | val 3.4783
step 106500: train 3.4632 | val 3.4881
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26262.48it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11277.61it/s]
step 106600: train 3.4705 | val 3.5554
step 106700: train 3.4191 | val 3.5075
step 106800: train 3.3282 | val 3.5369
step 106900: train 3.0797 | val 3.7552
step 107000: train 3.4363 | val 3.5383
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 107100: train 3.5834 | val 3.5220
step 107200: train 3.3520 | val 3.5621
step 107300: train 3.4512 | val 3.5358
step 107400: train 3.5899 | val 3.4295
step 107500: train 3.4336 | val 3.6082
step 107600: train 3.4906 | val 3.4168
step 107700: train 3.3550 | val 3.4432
step 107800: train 3.2071 | val 3.4903
step 107900: train 3.5542 | val 3.5193
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25499.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8308.61it/s]
step 108000: train 3.5873 | val 3.4290
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 108100: train 3.3889 | val 3.4904
step 108200: train 3.4172 | val 3.6302
step 108300: train 3.4607 | val 3.5230
step 108400: train 3.7620 | val 3.5280
step 108500: train 3.5058 | val 3.5551
step 108600: train 3.2559 | val 3.6124
step 108700: train 3.5251 | val 3.3778
step 108800: train 3.4643 | val 3.5433
step 108900: train 3.5344 | val 3.4942
step 109000: train 3.6044 | val 3.4709
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 109100: train 3.7747 | val 3.4505
step 109200: train 3.5649 | val 3.4460
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26323.08it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9215.94it/s]
step 109300: train 3.5431 | val 3.5086
step 109400: train 3.6027 | val 3.4560
step 109500: train 3.5119 | val 3.4854
step 109600: train 3.5922 | val 3.6914
step 109700: train 3.5320 | val 3.5148
step 109800: train 3.5219 | val 3.5071
step 109900: train 3.5012 | val 3.5534
step 110000: train 3.3637 | val 3.5166
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 110100: train 3.3849 | val 3.4336
step 110200: train 3.5419 | val 3.6153
step 110300: train 3.5165 | val 3.4104
step 110400: train 3.4880 | val 3.4578
step 110500: train 3.4512 | val 3.4861
step 110600: train 3.4974 | val 3.5249
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38597.69it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20356.46it/s]
step 110700: train 3.3030 | val 3.4433
step 110800: train 3.4677 | val 3.5014
step 110900: train 3.6159 | val 3.6284
step 111000: train 3.7187 | val 3.5478
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 111100: train 3.4611 | val 3.5408
step 111200: train 3.5576 | val 3.5743
step 111300: train 3.5362 | val 3.6236
step 111400: train 3.5398 | val 3.4161
step 111500: train 3.4278 | val 3.5582
step 111600: train 3.4254 | val 3.5152
step 111700: train 3.4817 | val 3.4949
step 111800: train 3.4595 | val 3.4572
step 111900: train 3.4019 | val 3.4569
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25567.07it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8594.51it/s]
step 112000: train 3.5481 | val 3.5254
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 112100: train 3.4891 | val 3.4788
step 112200: train 3.4339 | val 3.5001
step 112300: train 3.5005 | val 3.7428
step 112400: train 3.5677 | val 3.5214
step 112500: train 3.6299 | val 3.5220
step 112600: train 3.3815 | val 3.5872
step 112700: train 3.9135 | val 3.5583
step 112800: train 3.4294 | val 3.4888
step 112900: train 3.4664 | val 3.6394
step 113000: train 3.4348 | val 3.4486
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 113100: train 3.3988 | val 3.4806
step 113200: train 3.3565 | val 3.5163
step 113300: train 3.3501 | val 3.5683
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28550.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8657.87it/s]
step 113400: train 3.3746 | val 3.4788
step 113500: train 3.2412 | val 3.5432
step 113600: train 3.4271 | val 3.6617
step 113700: train 3.3004 | val 3.5747
step 113800: train 3.2923 | val 3.5771
step 113900: train 3.3048 | val 3.5957
step 114000: train 3.2205 | val 3.6714
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 114100: train 3.3200 | val 3.4422
step 114200: train 3.4432 | val 3.5908
step 114300: train 3.2290 | val 3.5234
step 114400: train 3.6515 | val 3.5032
step 114500: train 3.5087 | val 3.4524
step 114600: train 3.4987 | val 3.4531
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39172.52it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20296.66it/s]
step 114700: train 3.4956 | val 3.5098
step 114800: train 3.6843 | val 3.4619
step 114900: train 3.5674 | val 3.4901
step 115000: train 3.3679 | val 3.6844
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 115100: train 3.4592 | val 3.4999
step 115200: train 3.7210 | val 3.4977
step 115300: train 3.3480 | val 3.5417
step 115400: train 3.6193 | val 3.5062
step 115500: train 3.6613 | val 3.4203
step 115600: train 3.5129 | val 3.6052
step 115700: train 3.3712 | val 3.3783
step 115800: train 3.3854 | val 3.4361
step 115900: train 3.5495 | val 3.4480
step 116000: train 3.4403 | val 3.4997
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44473.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11649.23it/s]
step 116100: train 3.6449 | val 3.4129
step 116200: train 3.2664 | val 3.4614
step 116300: train 3.1568 | val 3.6327
step 116400: train 3.4236 | val 3.5027
step 116500: train 3.4799 | val 3.5123
step 116600: train 3.5606 | val 3.5341
step 116700: train 3.3809 | val 3.5946
step 116800: train 3.5406 | val 3.3769
step 116900: train 3.5572 | val 3.5524
step 117000: train 3.5225 | val 3.4727
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 117100: train 3.6721 | val 3.4452
step 117200: train 3.5724 | val 3.4259
step 117300: train 3.4782 | val 3.4269
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37982.46it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10427.29it/s]
step 117400: train 3.4652 | val 3.4956
step 117500: train 3.4696 | val 3.4542
step 117600: train 3.4579 | val 3.4646
step 117700: train 3.6657 | val 3.6890
step 117800: train 3.4984 | val 3.4992
step 117900: train 3.5174 | val 3.4957
step 118000: train 3.5514 | val 3.5376
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 118100: train 3.5485 | val 3.5149
step 118200: train 3.5134 | val 3.4275
step 118300: train 3.3718 | val 3.6288
step 118400: train 3.3882 | val 3.3854
step 118500: train 3.5898 | val 3.4492
step 118600: train 3.4624 | val 3.4635
step 118700: train 3.3690 | val 3.5189
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26708.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17899.24it/s]
step 118800: train 3.3440 | val 3.4318
step 118900: train 3.5716 | val 3.4976
step 119000: train 3.4642 | val 3.6204
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 119100: train 3.2389 | val 3.5216
step 119200: train 3.5003 | val 3.5371
step 119300: train 3.3339 | val 3.5604
step 119400: train 3.5329 | val 3.6333
step 119500: train 3.5484 | val 3.3968
step 119600: train 3.2144 | val 3.5706
step 119700: train 3.5200 | val 3.4901
step 119800: train 3.3710 | val 3.4813
step 119900: train 3.6210 | val 3.4460
step 120000: train 3.4056 | val 3.4524
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42016.44it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20597.09it/s]
step 120100: train 3.3280 | val 3.5217
step 120200: train 3.5948 | val 3.4649
step 120300: train 3.5500 | val 3.4973
step 120400: train 3.3279 | val 3.7155
step 120500: train 3.3856 | val 3.5169
step 120600: train 3.6787 | val 3.5135
step 120700: train 3.4176 | val 3.5698
step 120800: train 3.3127 | val 3.5536
step 120900: train 3.3003 | val 3.4561
step 121000: train 3.1269 | val 3.6348
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 121100: train 3.3849 | val 3.4231
step 121200: train 3.4201 | val 3.4710
step 121300: train 3.4526 | val 3.4972
step 121400: train 3.5605 | val 3.5525
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38691.51it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12273.27it/s]
step 121500: train 3.3058 | val 3.4576
step 121600: train 3.2598 | val 3.5231
step 121700: train 3.4095 | val 3.6541
step 121800: train 3.3892 | val 3.5621
step 121900: train 3.4006 | val 3.5598
step 122000: train 3.3297 | val 3.5822
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 122100: train 3.2136 | val 3.6685
step 122200: train 3.6502 | val 3.4289
step 122300: train 3.3645 | val 3.5684
step 122400: train 3.5480 | val 3.5042
step 122500: train 3.4930 | val 3.4786
step 122600: train 3.4068 | val 3.4327
step 122700: train 3.5350 | val 3.4323
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37458.72it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8545.23it/s]
step 122800: train 3.5197 | val 3.4988
step 122900: train 3.2987 | val 3.4371
step 123000: train 3.4273 | val 3.4754
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 123100: train 3.4971 | val 3.6903
step 123200: train 3.5097 | val 3.5008
step 123300: train 3.5765 | val 3.4887
step 123400: train 3.3232 | val 3.5291
step 123500: train 3.4026 | val 3.5027
step 123600: train 3.3880 | val 3.4080
step 123700: train 3.7607 | val 3.5996
step 123800: train 3.5689 | val 3.3874
step 123900: train 3.6014 | val 3.4237
step 124000: train 3.5464 | val 3.4573
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 124100: train 3.4835 | val 3.4827
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25971.72it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8516.72it/s]
step 124200: train 3.4374 | val 3.4008
step 124300: train 3.4721 | val 3.4619
step 124400: train 3.5281 | val 3.6005
step 124500: train 3.5864 | val 3.4966
step 124600: train 3.4235 | val 3.5124
step 124700: train 3.6418 | val 3.5382
step 124800: train 3.3375 | val 3.6053
step 124900: train 3.3884 | val 3.3862
step 125000: train 3.3502 | val 3.5384
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 125100: train 3.5217 | val 3.4704
step 125200: train 3.5051 | val 3.4611
step 125300: train 3.3732 | val 3.4232
step 125400: train 3.5363 | val 3.4223
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24224.31it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12372.32it/s]
step 125500: train 3.4867 | val 3.5026
step 125600: train 3.5803 | val 3.4462
step 125700: train 3.6125 | val 3.4768
step 125800: train 3.5780 | val 3.6779
step 125900: train 3.4461 | val 3.4861
step 126000: train 3.7583 | val 3.4856
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 126100: train 3.4302 | val 3.5500
step 126200: train 3.4926 | val 3.5062
step 126300: train 3.5506 | val 3.4190
step 126400: train 3.4441 | val 3.6005
step 126500: train 3.6516 | val 3.4021
step 126600: train 3.5166 | val 3.4417
step 126700: train 3.5371 | val 3.4690
step 126800: train 3.3276 | val 3.5142
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38298.20it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10239.11it/s]
step 126900: train 3.4915 | val 3.4275
step 127000: train 3.6369 | val 3.5009
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 127100: train 3.2972 | val 3.6241
step 127200: train 3.4843 | val 3.5186
step 127300: train 3.5198 | val 3.5291
step 127400: train 3.4046 | val 3.5527
step 127500: train 3.3985 | val 3.6371
step 127600: train 3.3399 | val 3.3937
step 127700: train 3.5613 | val 3.5547
step 127800: train 3.5082 | val 3.4877
step 127900: train 3.5235 | val 3.4785
step 128000: train 3.3613 | val 3.4397
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 128100: train 3.4742 | val 3.4479
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40170.06it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13297.16it/s]
step 128200: train 3.3818 | val 3.5175
step 128300: train 3.3613 | val 3.4714
step 128400: train 3.2841 | val 3.4968
step 128500: train 3.5181 | val 3.7152
step 128600: train 3.2517 | val 3.5172
step 128700: train 3.4775 | val 3.5166
step 128800: train 3.5118 | val 3.5735
step 128900: train 3.3956 | val 3.5485
step 129000: train 3.1795 | val 3.4572
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 129100: train 3.2545 | val 3.6181
step 129200: train 3.2882 | val 3.4311
step 129300: train 3.3836 | val 3.4720
step 129400: train 3.3250 | val 3.5014
step 129500: train 3.3845 | val 3.5511
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32372.89it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11184.38it/s]
step 129600: train 3.2376 | val 3.4599
step 129700: train 3.3289 | val 3.5251
step 129800: train 3.4137 | val 3.6580
step 129900: train 3.4618 | val 3.5373
step 130000: train 3.6049 | val 3.5318
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 130100: train 3.4037 | val 3.5441
step 130200: train 3.4636 | val 3.6175
step 130300: train 3.6380 | val 3.3895
step 130400: train 3.4480 | val 3.5213
step 130500: train 3.4197 | val 3.4845
step 130600: train 3.5179 | val 3.4607
step 130700: train 3.8043 | val 3.4226
step 130800: train 3.6029 | val 3.4411
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38027.76it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13708.80it/s]
step 130900: train 3.5367 | val 3.4773
step 131000: train 3.3854 | val 3.4382
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 131100: train 3.4060 | val 3.4503
step 131200: train 3.6113 | val 3.6678
step 131300: train 3.3348 | val 3.4790
step 131400: train 3.5610 | val 3.4819
step 131500: train 3.6835 | val 3.5192
step 131600: train 3.4062 | val 3.4913
step 131700: train 3.5170 | val 3.3977
step 131800: train 3.4707 | val 3.5914
step 131900: train 3.4432 | val 3.3916
step 132000: train 3.7005 | val 3.4227
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 132100: train 3.4137 | val 3.4614
step 132200: train 3.9200 | val 3.5065
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45495.67it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9442.83it/s]
step 132300: train 3.4285 | val 3.4163
step 132400: train 3.4452 | val 3.4679
step 132500: train 3.5995 | val 3.6116
step 132600: train 3.5845 | val 3.5103
step 132700: train 3.5584 | val 3.5057
step 132800: train 3.3438 | val 3.5356
step 132900: train 3.5108 | val 3.6109
step 133000: train 3.3106 | val 3.3760
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 133100: train 3.5124 | val 3.5274
step 133200: train 3.4682 | val 3.4768
step 133300: train 3.7712 | val 3.4666
step 133400: train 3.6394 | val 3.4323
step 133500: train 3.4643 | val 3.4363
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44733.80it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11226.08it/s]
step 133600: train 3.4108 | val 3.4901
step 133700: train 3.4820 | val 3.4470
step 133800: train 3.4591 | val 3.4602
step 133900: train 3.5163 | val 3.7049
step 134000: train 3.4161 | val 3.4985
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 134100: train 3.4408 | val 3.4944
step 134200: train 3.5752 | val 3.5489
step 134300: train 3.5003 | val 3.5232
step 134400: train 3.4171 | val 3.4396
step 134500: train 3.3703 | val 3.6002
step 134600: train 3.3280 | val 3.4018
step 134700: train 3.5022 | val 3.4393
step 134800: train 3.3750 | val 3.4756
step 134900: train 3.6724 | val 3.5144
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42670.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17058.93it/s]
step 135000: train 3.4006 | val 3.4345
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 135100: train 3.6518 | val 3.4981
step 135200: train 3.3599 | val 3.6276
step 135300: train 3.3329 | val 3.5268
step 135400: train 3.3157 | val 3.5330
step 135500: train 3.4744 | val 3.5635
step 135600: train 3.2961 | val 3.6312
step 135700: train 3.4897 | val 3.4139
step 135800: train 3.4295 | val 3.5520
step 135900: train 3.2787 | val 3.5151
step 136000: train 3.3134 | val 3.5006
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 136100: train 3.4672 | val 3.4577
step 136200: train 3.2837 | val 3.4547
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35884.64it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19686.95it/s]
step 136300: train 3.1439 | val 3.5252
step 136400: train 3.1622 | val 3.4741
step 136500: train 3.5530 | val 3.4980
step 136600: train 3.3430 | val 3.7193
step 136700: train 3.3543 | val 3.5226
step 136800: train 3.3703 | val 3.5201
step 136900: train 3.3460 | val 3.5713
step 137000: train 3.5452 | val 3.5501
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 137100: train 3.3204 | val 3.4592
step 137200: train 3.3483 | val 3.6172
step 137300: train 3.4770 | val 3.3964
step 137400: train 3.4751 | val 3.4326
step 137500: train 3.3838 | val 3.4576
step 137600: train 3.5687 | val 3.4999
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34445.62it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8514.25it/s]
step 137700: train 3.4478 | val 3.4138
step 137800: train 3.4040 | val 3.4731
step 137900: train 3.3002 | val 3.5988
step 138000: train 3.4665 | val 3.4915
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 138100: train 3.6368 | val 3.5133
step 138200: train 3.3126 | val 3.5242
step 138300: train 3.4184 | val 3.5879
step 138400: train 3.5737 | val 3.3715
step 138500: train 3.3105 | val 3.5069
step 138600: train 3.6017 | val 3.4718
step 138700: train 3.5280 | val 3.4295
step 138800: train 3.5234 | val 3.4004
step 138900: train 3.3068 | val 3.4132
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26127.87it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8706.52it/s]
step 139000: train 3.4347 | val 3.4763
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 139100: train 3.4040 | val 3.4188
step 139200: train 3.2187 | val 3.4365
step 139300: train 3.5649 | val 3.6505
step 139400: train 3.4244 | val 3.4717
step 139500: train 3.3663 | val 3.4781
step 139600: train 3.4035 | val 3.5163
step 139700: train 3.7445 | val 3.4820
step 139800: train 3.5914 | val 3.3881
step 139900: train 3.4372 | val 3.5817
step 140000: train 3.5977 | val 3.3565
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 140100: train 3.2912 | val 3.4040
step 140200: train 3.4616 | val 3.4409
step 140300: train 3.6360 | val 3.4876
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25659.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8727.61it/s]
step 140400: train 3.2751 | val 3.4020
step 140500: train 3.4754 | val 3.4558
step 140600: train 3.5619 | val 3.5883
step 140700: train 3.5784 | val 3.4941
step 140800: train 3.5576 | val 3.5082
step 140900: train 3.6170 | val 3.5323
step 141000: train 3.5188 | val 3.5920
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 141100: train 3.4979 | val 3.3767
step 141200: train 3.4921 | val 3.5145
step 141300: train 3.6590 | val 3.4593
step 141400: train 3.4214 | val 3.4412
step 141500: train 3.5960 | val 3.4125
step 141600: train 3.4011 | val 3.4133
Resolving data files: 100%
 2410/2410 [00:00<00:00, 29429.43it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14763.85it/s]
step 141700: train 3.4519 | val 3.4900
step 141800: train 3.5846 | val 3.4378
step 141900: train 3.5293 | val 3.4541
step 142000: train 3.2635 | val 3.6756
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 142100: train 3.4605 | val 3.4916
step 142200: train 3.5377 | val 3.4898
step 142300: train 3.3335 | val 3.5481
step 142400: train 3.3622 | val 3.5136
step 142500: train 3.3677 | val 3.4273
step 142600: train 3.4987 | val 3.5967
step 142700: train 3.4448 | val 3.3862
step 142800: train 3.3058 | val 3.4388
step 142900: train 3.3616 | val 3.4569
step 143000: train 3.4440 | val 3.5209
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36732.78it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14509.22it/s]
step 143100: train 3.3269 | val 3.4286
step 143200: train 3.6399 | val 3.4904
step 143300: train 3.4965 | val 3.6161
step 143400: train 3.3118 | val 3.5075
step 143500: train 3.1673 | val 3.5341
step 143600: train 3.4158 | val 3.5581
step 143700: train 3.2349 | val 3.6380
step 143800: train 3.2620 | val 3.4055
step 143900: train 3.5136 | val 3.5387
step 144000: train 3.3758 | val 3.4930
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 144100: train 3.3771 | val 3.4933
step 144200: train 3.3233 | val 3.4477
step 144300: train 3.4051 | val 3.4478
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26687.52it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8627.34it/s]
step 144400: train 3.2694 | val 3.5258
step 144500: train 3.2896 | val 3.4659
step 144600: train 3.5790 | val 3.4898
step 144700: train 3.3887 | val 3.7145
step 144800: train 3.5534 | val 3.5112
step 144900: train 3.4190 | val 3.5258
step 145000: train 3.4094 | val 3.5623
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 145100: train 3.1860 | val 3.5350
step 145200: train 3.2370 | val 3.4518
step 145300: train 3.7986 | val 3.6260
step 145400: train 3.4976 | val 3.4031
step 145500: train 3.6087 | val 3.4281
step 145600: train 3.4501 | val 3.4614
step 145700: train 3.4111 | val 3.5010
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43936.02it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13896.64it/s]
step 145800: train 3.4689 | val 3.4115
step 145900: train 3.3685 | val 3.4569
step 146000: train 3.4746 | val 3.5925
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 146100: train 3.3895 | val 3.4835
step 146200: train 3.7221 | val 3.5033
step 146300: train 3.5346 | val 3.5183
step 146400: train 3.3861 | val 3.5858
step 146500: train 3.5290 | val 3.3593
step 146900: train 3.4309 | val 3.3942
step 147000: train 3.5370 | val 3.4037
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40719.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17522.16it/s]
step 147100: train 3.4440 | val 3.4601
step 147200: train 3.2984 | val 3.3999
step 147300: train 3.5204 | val 3.4302
step 147400: train 3.4284 | val 3.6489
step 147500: train 3.3817 | val 3.4745
step 147600: train 3.7683 | val 3.4746
step 147700: train 3.2727 | val 3.5170
step 147800: train 3.5454 | val 3.4869
step 147900: train 3.5121 | val 3.3953
step 148000: train 3.5005 | val 3.5736
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 148100: train 3.4931 | val 3.3660
step 148200: train 3.3505 | val 3.4055
step 148300: train 3.4714 | val 3.4403
step 148400: train 3.4306 | val 3.4947
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25596.01it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17899.79it/s]
step 148500: train 3.4938 | val 3.4021
step 148600: train 3.4407 | val 3.4608
step 148700: train 3.4260 | val 3.5842
step 148800: train 3.2249 | val 3.4783
step 148900: train 3.5593 | val 3.4991
step 149000: train 3.3987 | val 3.5225
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 149100: train 3.6002 | val 3.5932
step 149200: train 3.4105 | val 3.3589
step 149300: train 3.3163 | val 3.5372
step 149400: train 3.4433 | val 3.4641
step 149500: train 3.4395 | val 3.4499
step 149600: train 3.4545 | val 3.4203
step 149700: train 3.3462 | val 3.4111
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26430.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8672.06it/s]
step 149800: train 3.3216 | val 3.4940
step 149900: train 3.5293 | val 3.4216
step 150000: train 4.0724 | val 3.4724
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 150100: train 3.5634 | val 3.6703
step 150200: train 3.5322 | val 3.4862
step 150300: train 3.3338 | val 3.4888
step 150400: train 3.6109 | val 3.5396
step 150500: train 3.4162 | val 3.5178
step 150600: train 3.3989 | val 3.4369
step 150700: train 3.4525 | val 3.5796
step 150800: train 3.5395 | val 3.3887
step 150900: train 3.6605 | val 3.4267
step 151000: train 3.4028 | val 3.4574
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 151100: train 3.2867 | val 3.5220
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26936.14it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 5602.28it/s]
step 151200: train 3.3260 | val 3.4297
step 151300: train 3.3141 | val 3.4966
step 151400: train 3.1509 | val 3.6203
step 151500: train 3.4992 | val 3.5299
step 151600: train 3.3943 | val 3.5351
step 151700: train 3.4386 | val 3.5586
step 151800: train 3.4041 | val 3.6231
step 151900: train 3.1601 | val 3.3988
step 152000: train 3.3055 | val 3.5548
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 152100: train 3.3514 | val 3.4860
step 152200: train 3.5328 | val 3.4846
step 152300: train 3.6242 | val 3.4457
step 152400: train 3.5532 | val 3.4415
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43467.47it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17575.13it/s]
step 152500: train 3.4934 | val 3.5244
step 152600: train 3.4169 | val 3.4692
step 152700: train 3.2605 | val 3.4867
step 152800: train 3.2256 | val 3.7091
step 152900: train 3.3469 | val 3.4945
step 153000: train 3.5085 | val 3.4902
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 153100: train 4.3151 | val 3.5326
step 153200: train 3.6272 | val 3.4868
step 153300: train 3.4725 | val 3.3990
step 153400: train 3.4086 | val 3.5787
step 153500: train 3.5608 | val 3.3772
step 153600: train 3.6022 | val 3.4141
step 153700: train 3.4172 | val 3.4424
step 153800: train 3.6340 | val 3.4792
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25379.43it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8479.70it/s]
step 153900: train 3.4842 | val 3.3951
step 154000: train 3.5936 | val 3.4430
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 154100: train 3.6734 | val 3.5781
step 154200: train 3.6159 | val 3.4753
step 154300: train 3.4301 | val 3.4883
step 154400: train 3.4123 | val 3.5144
step 154500: train 3.5100 | val 3.5707
step 154600: train 3.5275 | val 3.3540
step 154700: train 3.5286 | val 3.4977
step 154800: train 3.4431 | val 3.4505
step 154900: train 3.5226 | val 3.4334
step 155000: train 3.6089 | val 3.4107
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 155100: train 3.5006 | val 3.4078
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25862.81it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8464.55it/s]
step 155200: train 3.5673 | val 3.4769
step 155300: train 3.6712 | val 3.4230
step 155400: train 3.5713 | val 3.4425
step 155500: train 3.5323 | val 3.6569
step 155600: train 3.5881 | val 3.4748
step 155700: train 3.6403 | val 3.4752
step 155800: train 3.4514 | val 3.5177
step 155900: train 3.5853 | val 3.4806
step 156000: train 3.5574 | val 3.3867
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 156100: train 3.2504 | val 3.5720
step 156200: train 3.4522 | val 3.3656
step 156300: train 3.3587 | val 3.4066
step 156400: train 3.4553 | val 3.4412
step 156500: train 3.3902 | val 3.4947
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28551.38it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17171.17it/s]
step 156600: train 3.5109 | val 3.4104
step 156700: train 3.5058 | val 3.4655
step 156800: train 3.3577 | val 3.5977
step 156900: train 3.5678 | val 3.4972
step 157000: train 3.4340 | val 3.5115
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 157100: train 3.3964 | val 3.5264
step 157200: train 3.3058 | val 3.5990
step 157300: train 3.3575 | val 3.3768
step 157400: train 3.4953 | val 3.5347
step 157500: train 3.4539 | val 3.4635
step 157600: train 3.3904 | val 3.4616
step 157700: train 3.2919 | val 3.4230
step 157800: train 3.4305 | val 3.4285
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43499.45it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8773.65it/s]
step 157900: train 3.2599 | val 3.4930
step 158000: train 3.4649 | val 3.4427
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 158100: train 3.6006 | val 3.4579
step 158200: train 3.5178 | val 3.6918
step 158300: train 3.4434 | val 3.4962
step 158400: train 3.4442 | val 3.4868
step 158500: train 3.5983 | val 3.5546
step 158600: train 3.2629 | val 3.5200
step 158700: train 3.4452 | val 3.4344
step 158800: train 3.6058 | val 3.5964
step 158900: train 3.3064 | val 3.3959
step 159000: train 3.4117 | val 3.4438
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 159100: train 3.2727 | val 3.4689
step 159200: train 3.1416 | val 3.5242
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25347.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17601.47it/s]
step 159300: train 3.4532 | val 3.4314
step 159400: train 3.3964 | val 3.5047
step 159500: train 3.2267 | val 3.6295
step 159600: train 3.5556 | val 3.5332
step 159700: train 3.3208 | val 3.5333
step 159800: train 3.3049 | val 3.5561
step 159900: train 3.3609 | val 3.6026
step 160000: train 3.3922 | val 3.4017
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 160100: train 3.3705 | val 3.5464
step 160200: train 3.5214 | val 3.4801
step 160300: train 3.4455 | val 3.4531
step 160400: train 3.6734 | val 3.4138
step 160500: train 3.4086 | val 3.4127
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39057.64it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8241.67it/s]
step 160600: train 3.4815 | val 3.4764
step 160700: train 3.3392 | val 3.4247
step 160800: train 3.4053 | val 3.4442
step 160900: train 3.2584 | val 3.6536
step 161000: train 3.4795 | val 3.4657
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 161100: train 3.4366 | val 3.4648
step 161200: train 3.4246 | val 3.4969
step 161300: train 3.4443 | val 3.4531
step 161400: train 3.5778 | val 3.3694
step 161500: train 3.3224 | val 3.5642
step 161600: train 3.4861 | val 3.3357
step 161700: train 3.6368 | val 3.3943
step 161800: train 3.5899 | val 3.4215
step 161900: train 3.5729 | val 3.4659
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25389.38it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19153.32it/s]
step 162000: train 3.4836 | val 3.3912
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 162100: train 3.6321 | val 3.4307
step 162200: train 3.5209 | val 3.5731
step 162300: train 3.5240 | val 3.4656
step 162400: train 3.6027 | val 3.4767
step 162500: train 3.3953 | val 3.4991
step 162600: train 3.6564 | val 3.5581
step 162700: train 3.5413 | val 3.3392
step 162800: train 3.5925 | val 3.5122
step 162900: train 3.5646 | val 3.4336
step 163000: train 3.4688 | val 3.3998
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 163100: train 3.1724 | val 3.3818
step 163200: train 3.4789 | val 3.3950
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44407.57it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8615.44it/s]
step 163300: train 3.4358 | val 3.4581
step 163400: train 3.4076 | val 3.4169
step 163500: train 3.4001 | val 3.4291
step 163600: train 3.4116 | val 3.6521
step 163700: train 3.3968 | val 3.4579
step 163800: train 3.5257 | val 3.4590
step 163900: train 3.4854 | val 3.5094
step 164000: train 3.4325 | val 3.4743
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 164100: train 3.8337 | val 3.3873
step 164200: train 3.3841 | val 3.5676
step 164300: train 3.5467 | val 3.3612
step 164400: train 3.4520 | val 3.3998
step 164500: train 3.4593 | val 3.4280
step 164600: train 3.4320 | val 3.5001
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25776.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8443.13it/s]
step 164700: train 3.4860 | val 3.4032
step 164800: train 3.2547 | val 3.4630
step 164900: train 3.2366 | val 3.5969
step 165000: train 3.4598 | val 3.4875
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 165100: train 3.5902 | val 3.5035
step 165200: train 3.2411 | val 3.5230
step 165300: train 3.4775 | val 3.5932
step 165400: train 3.4611 | val 3.3718
step 165500: train 3.5754 | val 3.5262
step 165600: train 3.4615 | val 3.4594
step 165700: train 3.3251 | val 3.4490
step 165800: train 3.5092 | val 3.4065
step 165900: train 3.4537 | val 3.4178
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27481.65it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8646.14it/s]
step 166000: train 3.4725 | val 3.4894
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 166100: train 3.2600 | val 3.4348
step 166200: train 3.4963 | val 3.4532
step 166300: train 4.2063 | val 3.6897
step 166400: train 3.3959 | val 3.4856
step 166500: train 3.3994 | val 3.4825
step 166600: train 3.2970 | val 3.5457
step 166700: train 3.3450 | val 3.5183
step 166800: train 3.4530 | val 3.4281
step 166900: train 3.4368 | val 3.5989
step 167000: train 3.4411 | val 3.3904
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 167100: train 3.1249 | val 3.4320
step 167200: train 3.5761 | val 3.4629
step 167300: train 3.3929 | val 3.5239
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26726.19it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8441.91it/s]
step 167400: train 3.3591 | val 3.4306
step 167500: train 3.3325 | val 3.4964
step 167600: train 3.2331 | val 3.6225
step 167700: train 3.2673 | val 3.5282
step 167800: train 3.0838 | val 3.5219
step 167900: train 3.3196 | val 3.5463
step 168000: train 3.3175 | val 3.6215
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 168100: train 3.4180 | val 3.3991
step 168200: train 3.3109 | val 3.5315
step 168300: train 3.4631 | val 3.4661
step 168400: train 3.4206 | val 3.4407
step 168500: train 3.5641 | val 3.3958
step 168600: train 3.4555 | val 3.4020
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43872.33it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17102.16it/s]
step 168700: train 3.4362 | val 3.4642
step 168800: train 3.5469 | val 3.4105
step 168900: train 3.3087 | val 3.4314
step 169000: train 3.4537 | val 3.6498
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 169100: train 3.3736 | val 3.4607
step 169200: train 3.2876 | val 3.4544
step 169300: train 3.3858 | val 3.5003
step 169400: train 3.5198 | val 3.4687
step 169500: train 3.7697 | val 3.3713
step 169600: train 3.3832 | val 3.5560
step 169700: train 3.2880 | val 3.3553
step 169800: train 3.6111 | val 3.3921
step 169900: train 3.5419 | val 3.4202
step 170000: train 3.3582 | val 3.4616
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27060.46it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 5585.92it/s]
step 170100: train 3.4582 | val 3.3697
step 170200: train 3.6224 | val 3.4359
step 170300: train 3.5772 | val 3.5693
step 170400: train 3.8836 | val 3.4689
step 170500: train 3.4384 | val 3.4782
step 170600: train 3.3106 | val 3.4987
step 170700: train 3.5677 | val 3.5663
step 170800: train 3.5565 | val 3.3404
step 170900: train 3.5494 | val 3.4847
step 171000: train 3.5842 | val 3.4400
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 171100: train 3.5474 | val 3.4306
step 171200: train 3.4445 | val 3.4014
step 171300: train 3.3018 | val 3.3997
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25890.70it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8433.06it/s]
step 171400: train 3.1592 | val 3.4585
step 171500: train 3.3996 | val 3.4171
step 171600: train 3.4187 | val 3.4374
step 171700: train 3.5614 | val 3.6500
step 171800: train 3.3955 | val 3.4564
step 171900: train 3.7586 | val 3.4609
step 172000: train 3.4901 | val 3.5058
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 172100: train 3.5405 | val 3.4663
step 172200: train 3.3664 | val 3.3981
step 172300: train 3.4546 | val 3.5641
step 172400: train 3.4723 | val 3.3684
step 172500: train 3.4942 | val 3.4110
step 172600: train 3.4872 | val 3.4379
step 172700: train 3.4067 | val 3.4861
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39456.31it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16573.60it/s]
step 172800: train 3.4170 | val 3.3968
step 172900: train 3.4225 | val 3.4622
step 173000: train 3.3511 | val 3.5808
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 173100: train 3.3964 | val 3.4889
step 173200: train 3.1644 | val 3.4911
step 173300: train 3.3529 | val 3.5142
step 173400: train 3.5045 | val 3.5897
step 173500: train 3.5034 | val 3.3636
step 173600: train 3.2845 | val 3.5062
step 173700: train 3.4349 | val 3.4586
step 173800: train 3.5052 | val 3.4450
step 173900: train 3.2154 | val 3.4197
step 174000: train 3.4308 | val 3.4125
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26249.53it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14696.60it/s]
step 174100: train 3.3233 | val 3.4864
step 174200: train 3.2126 | val 3.4574
step 174300: train 3.2360 | val 3.4559
step 174400: train 3.3235 | val 3.6870
step 174500: train 3.7804 | val 3.4869
step 174600: train 3.3517 | val 3.4927
step 174700: train 3.1797 | val 3.5394
step 174800: train 3.2626 | val 3.5040
step 174900: train 3.4639 | val 3.4270
step 175000: train 3.2597 | val 3.5867
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 175100: train 3.3244 | val 3.3894
step 175200: train 3.1123 | val 3.4363
step 175300: train 3.2895 | val 3.4588
step 175400: train 3.4617 | val 3.5136
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26118.83it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15716.57it/s]
step 175500: train 3.3016 | val 3.4268
step 175600: train 3.4768 | val 3.4942
step 175700: train 3.4542 | val 3.6204
step 175800: train 3.5705 | val 3.5052
step 175900: train 3.5050 | val 3.4977
step 176000: train 3.5370 | val 3.5176
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 176100: train 3.6162 | val 3.5723
step 176200: train 3.3493 | val 3.3525
step 176300: train 3.2261 | val 3.4915
step 176400: train 3.4801 | val 3.4421
step 176500: train 3.4387 | val 3.4243
step 176600: train 3.4965 | val 3.4121
step 176700: train 3.4836 | val 3.3935
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39491.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19657.29it/s]
step 176800: train 3.6581 | val 3.4623
step 176900: train 3.4039 | val 3.4116
step 177000: train 3.4657 | val 3.4240
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 177100: train 3.5179 | val 3.6364
step 177200: train 3.4012 | val 3.4596
step 177300: train 3.3265 | val 3.4470
step 177400: train 3.2915 | val 3.4939
step 177500: train 3.5338 | val 3.4512
step 177600: train 3.4971 | val 3.3751
step 177700: train 3.5621 | val 3.5463
step 177800: train 3.4887 | val 3.3539
step 177900: train 3.5714 | val 3.3960
step 178000: train 3.6760 | val 3.4299
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 178100: train 3.4025 | val 3.4727
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37004.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15870.77it/s]
step 178200: train 3.4754 | val 3.3874
step 178300: train 3.5087 | val 3.4377
step 178400: train 3.3031 | val 3.5708
step 178500: train 3.3019 | val 3.4711
step 178600: train 3.4080 | val 3.4779
step 178700: train 3.5529 | val 3.5204
step 178800: train 3.4343 | val 3.5614
step 178900: train 3.3784 | val 3.3518
step 179000: train 3.3526 | val 3.4857
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 179100: train 3.5193 | val 3.4409
step 179200: train 3.4114 | val 3.4222
step 179300: train 3.2091 | val 3.4043
step 179400: train 3.4796 | val 3.3952
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41277.29it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14378.12it/s]
step 179500: train 3.4831 | val 3.4710
step 179600: train 3.5254 | val 3.4227
step 179700: train 3.3587 | val 3.4320
step 179800: train 3.4512 | val 3.6414
step 179900: train 3.3985 | val 3.4610
step 180000: train 3.4981 | val 3.4607
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 180100: train 3.4752 | val 3.5225
step 180200: train 3.3826 | val 3.4765
step 180300: train 3.4596 | val 3.4186
step 180400: train 3.2407 | val 3.5661
step 180500: train 3.2991 | val 3.3641
step 180600: train 3.6160 | val 3.4138
step 180700: train 3.5721 | val 3.4324
step 180800: train 3.3972 | val 3.4872
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38669.75it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20943.10it/s]
step 180900: train 3.4823 | val 3.3995
step 181000: train 3.3826 | val 3.4643
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 181100: train 3.5072 | val 3.5874
step 181200: train 3.5032 | val 3.4940
step 181300: train 3.5055 | val 3.5022
step 181400: train 3.3973 | val 3.5230
step 181500: train 3.4200 | val 3.6018
step 181600: train 3.3073 | val 3.3834
step 181700: train 3.3430 | val 3.5166
step 181800: train 3.2678 | val 3.4649
step 181900: train 3.5014 | val 3.4597
step 182000: train 3.3285 | val 3.4268
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 182100: train 3.3519 | val 3.4222
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42492.09it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 21501.38it/s]
step 182200: train 3.1199 | val 3.4926
step 182300: train 3.4390 | val 3.4379
step 182400: train 3.4019 | val 3.4652
step 182500: train 3.2046 | val 3.6918
step 182600: train 3.2138 | val 3.4933
step 182700: train 3.2289 | val 3.4919
step 182800: train 3.3620 | val 3.5376
step 182900: train 3.4022 | val 3.5162
step 183000: train 3.3490 | val 3.4231
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 183100: train 3.4906 | val 3.5811
step 183200: train 3.5315 | val 3.3704
step 183300: train 3.8460 | val 3.3917
step 183400: train 3.4209 | val 3.4330
step 183500: train 3.5367 | val 3.4898
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39982.57it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 21159.69it/s]
step 183600: train 3.4656 | val 3.3789
step 183700: train 3.2085 | val 3.4329
step 183800: train 3.4146 | val 3.5751
step 183900: train 3.4036 | val 3.4627
step 184000: train 3.4618 | val 3.4797
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 184100: train 3.5658 | val 3.4863
step 184200: train 3.6413 | val 3.5495
step 184300: train 3.2469 | val 3.3360
step 184400: train 3.4713 | val 3.4798
step 184500: train 3.6212 | val 3.4236
step 184600: train 3.5487 | val 3.3933
step 184700: train 3.5762 | val 3.3605
step 184800: train 3.4911 | val 3.3776
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43581.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17670.86it/s]
step 184900: train 3.1970 | val 3.4281
step 185000: train 3.4688 | val 3.3889
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 185100: train 3.3649 | val 3.4051
step 185200: train 3.4051 | val 3.6256
step 185300: train 3.4085 | val 3.4396
step 185400: train 3.4365 | val 3.4455
step 185500: train 3.6114 | val 3.4733
step 185600: train 3.4200 | val 3.4397
step 185700: train 3.5607 | val 3.3649
step 185800: train 3.4962 | val 3.5507
step 185900: train 3.3155 | val 3.3278
step 186000: train 3.5971 | val 3.3815
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 186100: train 3.6756 | val 3.4072
step 186200: train 3.3880 | val 3.4576
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38813.63it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18658.53it/s]
step 186300: train 3.3847 | val 3.3710
step 186400: train 3.4281 | val 3.4221
step 186500: train 3.5372 | val 3.5581
step 186600: train 3.3340 | val 3.4667
step 186700: train 3.4831 | val 3.4739
step 186800: train 3.3575 | val 3.4914
step 186900: train 3.4746 | val 3.5630
step 187000: train 3.5620 | val 3.3451
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 187100: train 3.4163 | val 3.4874
step 187200: train 3.0084 | val 3.4281
step 187300: train 3.4496 | val 3.4171
step 187400: train 3.4559 | val 3.3873
step 187500: train 3.3358 | val 3.3917
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39545.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20136.57it/s]
step 187600: train 3.6360 | val 3.4536
step 187700: train 3.6727 | val 3.4091
step 187800: train 3.4973 | val 3.4317
step 187900: train 3.3396 | val 3.6505
step 188000: train 3.3827 | val 3.4596
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 188100: train 3.4270 | val 3.4629
step 188200: train 3.3292 | val 3.5049
step 188300: train 3.2943 | val 3.4775
step 188400: train 3.4131 | val 3.3918
step 188500: train 3.4306 | val 3.5532
step 188600: train 3.4381 | val 3.3521
step 188700: train 3.4044 | val 3.4033
step 188800: train 3.4778 | val 3.4317
step 188900: train 3.1574 | val 3.4858
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44834.39it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20362.11it/s]
step 189000: train 3.3260 | val 3.3909
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 189100: train 3.4155 | val 3.4527
step 189200: train 3.5242 | val 3.5817
step 189300: train 3.3097 | val 3.4785
step 189400: train 3.0659 | val 3.4946
step 189500: train 3.3267 | val 3.5162
step 189600: train 3.3353 | val 3.5866
step 189700: train 3.2973 | val 3.3711
step 189800: train 3.4138 | val 3.5232
step 189900: train 3.2576 | val 3.4568
step 190000: train 3.4505 | val 3.4475
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 190100: train 3.3794 | val 3.4115
step 190200: train 3.5907 | val 3.4214
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37594.13it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20949.82it/s]
step 190300: train 3.6847 | val 3.4933
step 190400: train 3.3100 | val 3.4388
step 190500: train 3.2365 | val 3.4557
step 190600: train 3.3611 | val 3.6885
step 190700: train 3.3729 | val 3.4782
step 190800: train 3.2694 | val 3.4823
step 190900: train 3.3923 | val 3.5221
step 191000: train 3.3530 | val 3.5071
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 191100: train 3.3035 | val 3.4166
step 191200: train 3.6000 | val 3.5727
step 191300: train 3.5006 | val 3.3531
step 191400: train 3.4911 | val 3.4006
step 191500: train 3.4987 | val 3.4157
step 191600: train 3.4758 | val 3.4515
Resolving data files: 100%
 2410/2410 [00:00<00:00, 33102.91it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18134.17it/s]
step 191700: train 3.2650 | val 3.3674
step 191800: train 3.4719 | val 3.4312
step 191900: train 3.3412 | val 3.5471
step 192000: train 3.4243 | val 3.4592
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 192100: train 3.3608 | val 3.4638
step 192200: train 3.5882 | val 3.4966
step 192300: train 3.3756 | val 3.5416
step 192400: train 3.3845 | val 3.3232
step 192500: train 3.5093 | val 3.4666
step 192600: train 3.4751 | val 3.4180
step 192700: train 3.3107 | val 3.3951
step 192800: train 3.4055 | val 3.3587
step 192900: train 3.4028 | val 3.3651
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44697.40it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18918.83it/s]
step 193000: train 3.5905 | val 3.4200
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 193100: train 3.6432 | val 3.3845
step 193200: train 3.5538 | val 3.4028
step 193300: train 3.4127 | val 3.6229
step 193400: train 3.4332 | val 3.4396
step 193500: train 3.5491 | val 3.4409
step 193600: train 3.5968 | val 3.4753
step 193700: train 3.3812 | val 3.4538
step 193800: train 3.4320 | val 3.3549
step 193900: train 3.3930 | val 3.5515
step 194000: train 3.6321 | val 3.3340
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 194100: train 3.5001 | val 3.3809
step 194200: train 3.4583 | val 3.4029
step 194300: train 3.3810 | val 3.4580
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35382.33it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19685.63it/s]
step 194400: train 3.3961 | val 3.3652
step 194500: train 3.4922 | val 3.4259
step 194600: train 3.4223 | val 3.5565
step 194700: train 3.4310 | val 3.4546
step 194800: train 3.4809 | val 3.4633
step 194900: train 3.4650 | val 3.4864
step 195000: train 3.4332 | val 3.5563
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 195100: train 3.5609 | val 3.3361
step 195200: train 3.3914 | val 3.4930
step 195300: train 3.4322 | val 3.4369
step 195400: train 3.3427 | val 3.4211
step 195500: train 3.4981 | val 3.3845
step 195600: train 3.6251 | val 3.3949
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41425.48it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 21054.99it/s]
step 195700: train 3.3078 | val 3.4532
step 195800: train 3.5152 | val 3.4071
step 195900: train 3.3061 | val 3.4279
step 196000: train 3.5388 | val 3.6465
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 196100: train 3.4067 | val 3.4539
step 196200: train 3.6233 | val 3.4527
step 196300: train 3.3285 | val 3.5047
step 196400: train 3.3986 | val 3.4603
step 196500: train 3.4217 | val 3.3801
step 196600: train 3.4742 | val 3.5496
step 196700: train 3.5004 | val 3.3497
step 196800: train 3.4727 | val 3.3979
step 196900: train 3.1906 | val 3.4180
step 197000: train 3.1140 | val 3.4886
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34422.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18175.71it/s]
step 197100: train 3.5629 | val 3.3945
step 197200: train 3.2398 | val 3.4564
step 197300: train 3.4594 | val 3.5871
step 197400: train 3.5433 | val 3.4941
step 197500: train 3.3482 | val 3.5038
step 197600: train 3.1326 | val 3.5209
step 197700: train 3.1911 | val 3.5922
step 197800: train 3.3081 | val 3.3738
step 197900: train 3.0463 | val 3.5176
step 198000: train 3.4117 | val 3.4502
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 198100: train 3.4180 | val 3.4565
step 198200: train 3.3960 | val 3.4099
step 198300: train 3.1909 | val 3.4071
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42152.93it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17703.35it/s]
step 198400: train 3.2699 | val 3.4751
step 198500: train 3.2641 | val 3.4340
step 198600: train 3.4216 | val 3.4546
step 198700: train 3.0074 | val 3.6686
step 198800: train 3.3851 | val 3.4552
step 198900: train 3.3612 | val 3.4342
step 199000: train 3.4287 | val 3.4807
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 199100: train 3.3171 | val 3.4489
step 199200: train 3.4827 | val 3.3602
step 199300: train 3.5079 | val 3.5394
step 199400: train 3.5039 | val 3.3313
step 199500: train 3.6961 | val 3.3626
step 199600: train 3.3414 | val 3.4029
step 199700: train 3.3986 | val 3.4273
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36131.56it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16628.98it/s]
step 199800: train 3.4699 | val 3.3532
step 199900: train 3.5819 | val 3.4051
step 200000: train 3.3530 | val 3.5535
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 200100: train 3.3707 | val 3.4423
step 200200: train 3.2976 | val 3.4588
step 200300: train 3.3634 | val 3.4705
step 200400: train 3.5682 | val 3.5400
step 200500: train 3.3919 | val 3.3079
step 200600: train 3.2832 | val 3.4717
step 200700: train 3.5347 | val 3.4122
step 200800: train 3.6167 | val 3.3979
step 200900: train 3.5682 | val 3.3631
step 201000: train 3.3914 | val 3.3712
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41919.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12659.59it/s]
step 201100: train 3.5172 | val 3.4286
step 201200: train 3.3668 | val 3.3797
step 201300: train 3.4801 | val 3.4000
step 201400: train 3.3823 | val 3.6205
step 201500: train 3.4051 | val 3.4257
step 201600: train 3.9547 | val 3.4317
step 201700: train 3.4618 | val 3.4840
step 201800: train 3.4759 | val 3.4412
step 201900: train 3.3816 | val 3.3568
step 202000: train 3.3741 | val 3.5303
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 202100: train 3.5040 | val 3.3287
step 202200: train 3.4749 | val 3.3750
step 202300: train 3.4876 | val 3.4035
step 202400: train 3.5249 | val 3.4463
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36800.98it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14762.36it/s]
step 202500: train 3.5388 | val 3.3656
step 202600: train 3.4475 | val 3.4328
step 202700: train 3.4626 | val 3.5608
step 202800: train 3.3865 | val 3.4609
step 202900: train 3.3709 | val 3.4738
step 203000: train 3.4130 | val 3.4987
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 203100: train 3.4264 | val 3.5663
step 203200: train 3.5676 | val 3.3395
step 203300: train 3.4510 | val 3.5029
step 203400: train 3.4904 | val 3.4321
step 203500: train 3.5427 | val 3.4253
step 203600: train 3.3334 | val 3.3875
step 203700: train 3.4330 | val 3.3888
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34752.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9817.80it/s]
step 203800: train 3.4075 | val 3.4529
step 203900: train 3.5230 | val 3.4057
step 204000: train 3.3788 | val 3.4259
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 204100: train 3.4595 | val 3.6460
step 204200: train 3.3813 | val 3.4498
step 204300: train 3.2325 | val 3.4532
step 204400: train 3.2743 | val 3.5064
step 204500: train 3.3200 | val 3.4771
step 204600: train 3.3945 | val 3.4007
step 204700: train 3.1215 | val 3.5830
step 204800: train 3.2072 | val 3.3658
step 204900: train 3.2018 | val 3.4042
step 205000: train 3.2245 | val 3.4334
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 205100: train 3.3880 | val 3.4960
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42488.16it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15472.24it/s]
step 205200: train 3.2944 | val 3.3968
step 205300: train 3.3299 | val 3.4692
step 205400: train 3.2513 | val 3.6000
step 205500: train 3.3476 | val 3.4911
step 205600: train 3.2208 | val 3.4985
step 205700: train 3.3486 | val 3.5218
step 205800: train 3.4589 | val 3.5917
step 205900: train 3.2126 | val 3.3687
step 206000: train 3.2193 | val 3.5104
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 206100: train 3.6252 | val 3.4371
step 206200: train 3.2906 | val 3.4162
step 206300: train 3.4898 | val 3.3836
step 206400: train 3.1480 | val 3.3786
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42849.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15406.88it/s]
step 206500: train 3.3563 | val 3.4427
step 206600: train 3.6689 | val 3.3933
step 206700: train 3.4625 | val 3.4072
step 206800: train 3.3088 | val 3.6274
step 206900: train 3.4693 | val 3.4200
step 207000: train 3.3270 | val 3.4294
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 207100: train 3.2724 | val 3.4710
step 207200: train 3.4906 | val 3.4395
step 207300: train 3.5079 | val 3.3429
step 207400: train 3.3873 | val 3.5172
step 207500: train 3.4597 | val 3.3066
step 207600: train 3.4471 | val 3.3606
step 207700: train 3.7026 | val 3.3882
step 207800: train 3.5179 | val 3.4313
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39793.22it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15146.19it/s]
step 207900: train 3.3188 | val 3.3349
step 208000: train 3.8154 | val 3.3971
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 208100: train 3.4422 | val 3.5385
step 208200: train 3.4730 | val 3.4308
step 208300: train 3.5047 | val 3.4515
step 208400: train 3.2546 | val 3.4629
step 208500: train 3.5167 | val 3.5210
step 208600: train 3.3204 | val 3.3031
step 208700: train 3.4212 | val 3.4605
step 208800: train 3.5006 | val 3.4070
step 208900: train 3.4277 | val 3.3699
step 209000: train 3.6626 | val 3.3553
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 209100: train 3.3926 | val 3.3640
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24364.74it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9447.09it/s]
step 209200: train 3.6180 | val 3.4235
step 209300: train 3.4140 | val 3.3687
step 209400: train 3.4435 | val 3.3986
step 209500: train 2.8663 | val 3.6156
step 209600: train 3.4811 | val 3.4267
step 209700: train 3.3235 | val 3.4258
step 209800: train 3.5371 | val 3.4803
step 209900: train 3.5080 | val 3.4353
step 210000: train 3.3686 | val 3.3425
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 210100: train 3.3966 | val 3.5169
step 210200: train 3.5307 | val 3.3141
step 210300: train 3.4525 | val 3.3650
step 210400: train 3.3001 | val 3.3985
step 210500: train 3.2892 | val 3.4429
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25750.81it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8327.93it/s]
step 210600: train 3.3644 | val 3.3578
step 210700: train 3.3045 | val 3.4223
step 210800: train 3.4501 | val 3.5607
step 210900: train 3.3223 | val 3.4642
step 211000: train 3.3441 | val 3.4677
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 211100: train 3.3513 | val 3.4889
step 211200: train 3.7162 | val 3.5554
step 211300: train 3.3649 | val 3.3419
step 211400: train 3.5882 | val 3.4936
step 211500: train 3.1751 | val 3.4241
step 211600: train 3.3399 | val 3.4057
step 211700: train 3.3906 | val 3.3733
step 211800: train 3.3901 | val 3.3759
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25889.90it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9822.56it/s]
step 211900: train 3.5903 | val 3.4488
step 212000: train 3.4126 | val 3.3997
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 212100: train 3.3584 | val 3.4193
step 212200: train 3.3087 | val 3.6422
step 212300: train 3.5970 | val 3.4436
step 212400: train 3.3553 | val 3.4583
step 212500: train 3.2108 | val 3.5050
step 212600: train 3.3269 | val 3.4741
step 212700: train 3.2280 | val 3.3863
step 212800: train 3.3732 | val 3.5572
step 212900: train 3.3230 | val 3.3503
step 213000: train 3.3791 | val 3.4002
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 213100: train 3.3440 | val 3.4295
step 213200: train 3.5539 | val 3.4786
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45881.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8619.61it/s]
step 213300: train 3.3839 | val 3.3851
step 213400: train 3.3377 | val 3.4522
step 213500: train 3.2937 | val 3.5953
step 213600: train 2.9525 | val 3.4859
step 213700: train 3.4486 | val 3.4985
step 213800: train 3.3439 | val 3.5171
step 213900: train 3.4183 | val 3.5871
step 214000: train 3.3380 | val 3.3550
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 214100: train 3.1955 | val 3.4936
step 214200: train 3.5667 | val 3.4290
step 214300: train 3.5013 | val 3.4030
step 214400: train 3.8082 | val 3.3764
step 214500: train 3.6423 | val 3.3726
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26526.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10884.20it/s]
step 214600: train 3.5412 | val 3.4327
step 214700: train 3.4219 | val 3.3809
step 214800: train 3.5441 | val 3.3954
step 214900: train 3.4496 | val 3.6249
step 215000: train 3.4654 | val 3.4210
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 215100: train 3.4214 | val 3.4233
step 215200: train 3.3828 | val 3.4593
step 215300: train 3.3774 | val 3.4439
step 215400: train 3.4619 | val 3.3357
step 215500: train 3.4544 | val 3.5251
step 215600: train 3.6343 | val 3.3202
step 215700: train 3.4107 | val 3.3557
step 215800: train 3.4305 | val 3.3704
step 215900: train 3.1793 | val 3.4321
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34552.05it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 19032.88it/s]
step 216000: train 3.4156 | val 3.3364
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 216100: train 3.4033 | val 3.3975
step 216200: train 3.5676 | val 3.5251
step 216300: train 3.4278 | val 3.4363
step 216400: train 3.4842 | val 3.4512
step 216500: train 3.5875 | val 3.4660
step 216600: train 3.1854 | val 3.5283
step 216700: train 3.5753 | val 3.3144
step 216800: train 3.5278 | val 3.4834
step 216900: train 3.3436 | val 3.4070
step 217000: train 3.5219 | val 3.3908
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 217100: train 3.4865 | val 3.3663
step 217200: train 3.5678 | val 3.3465
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41062.83it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 21031.61it/s]
step 217300: train 3.5406 | val 3.4354
step 217400: train 3.4543 | val 3.3671
step 217500: train 3.6274 | val 3.3926
step 217600: train 3.4724 | val 3.6367
step 217700: train 3.4963 | val 3.4301
step 217800: train 3.5970 | val 3.4259
step 217900: train 3.5345 | val 3.4660
step 218000: train 3.4297 | val 3.4345
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 218100: train 3.5490 | val 3.3532
step 218200: train 3.5362 | val 3.5309
step 218300: train 3.5160 | val 3.3258
step 218400: train 3.3410 | val 3.3736
step 218500: train 3.5306 | val 3.3957
step 218600: train 3.4541 | val 3.4425
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25921.24it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7320.72it/s]
step 218700: train 3.2138 | val 3.3583
step 218800: train 3.1342 | val 3.4235
step 218900: train 3.3658 | val 3.5499
step 219000: train 3.5463 | val 3.4600
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 219100: train 3.3997 | val 3.4627
step 219200: train 3.7507 | val 3.4879
step 219300: train 3.3782 | val 3.5552
step 219400: train 3.3079 | val 3.3246
step 219500: train 3.4726 | val 3.4652
step 219600: train 3.1054 | val 3.4310
step 219700: train 3.4335 | val 3.4081
step 219800: train 3.4948 | val 3.3773
step 219900: train 3.3052 | val 3.3747
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31896.45it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14080.58it/s]
step 220000: train 3.4918 | val 3.4537
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 220100: train 3.3839 | val 3.4217
step 220200: train 3.3452 | val 3.4204
step 220300: train 3.4542 | val 3.6446
step 220400: train 3.2849 | val 3.4586
step 220500: train 3.4573 | val 3.4434
step 220600: train 3.2792 | val 3.4961
step 220700: train 3.3969 | val 3.4752
step 220800: train 3.3409 | val 3.3806
step 220900: train 3.4947 | val 3.5446
step 221000: train 3.1981 | val 3.3555
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 221100: train 3.2867 | val 3.3871
step 221200: train 3.2292 | val 3.4134
step 221300: train 3.3044 | val 3.4674
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24307.79it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13670.18it/s]
step 221400: train 3.2003 | val 3.3817
step 221500: train 3.3595 | val 3.4438
step 221600: train 3.4294 | val 3.5891
step 221700: train 3.2769 | val 3.4697
step 221800: train 3.3486 | val 3.4610
step 221900: train 3.5874 | val 3.4695
step 222000: train 3.4835 | val 3.5416
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 222100: train 3.6463 | val 3.3150
step 222200: train 3.4736 | val 3.4412
step 222300: train 3.5493 | val 3.4106
step 222400: train 3.5760 | val 3.3844
step 222500: train 3.6896 | val 3.3493
step 222600: train 3.2584 | val 3.3530
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25286.17it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12588.76it/s]
step 222700: train 3.4146 | val 3.4132
step 222800: train 3.5706 | val 3.3556
step 222900: train 3.5206 | val 3.3787
step 223000: train 3.3377 | val 3.5989
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 223100: train 3.2920 | val 3.4075
step 223200: train 3.1970 | val 3.4102
step 223300: train 3.4088 | val 3.4501
step 223400: train 3.2581 | val 3.4196
step 223500: train 3.5229 | val 3.3392
step 223600: train 3.4477 | val 3.5121
step 223700: train 3.1888 | val 3.3128
step 223800: train 3.1809 | val 3.3507
step 223900: train 3.6403 | val 3.3898
step 224000: train 3.3767 | val 3.4243
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25445.88it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8808.39it/s]
step 224100: train 3.3404 | val 3.3367
step 224200: train 3.4465 | val 3.3927
step 224300: train 3.3980 | val 3.5247
step 224400: train 3.4633 | val 3.4247
step 224500: train 3.3257 | val 3.4379
step 224600: train 3.5162 | val 3.4691
step 224700: train 3.5654 | val 3.5420
step 224800: train 3.3213 | val 3.3118
step 224900: train 3.3778 | val 3.4471
step 225000: train 3.4115 | val 3.4050
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 225100: train 3.3773 | val 3.3944
step 225200: train 3.3634 | val 3.3529
step 225300: train 3.2776 | val 3.3533
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26339.06it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16915.93it/s]
step 225400: train 3.1920 | val 3.4273
step 225500: train 3.3118 | val 3.3644
step 225600: train 3.4824 | val 3.3901
step 225700: train 3.2353 | val 3.6179
step 225800: train 3.3014 | val 3.4196
step 225900: train 3.3891 | val 3.4303
step 226000: train 3.4944 | val 3.4815
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 226100: train 3.5148 | val 3.4492
step 226200: train 3.5095 | val 3.3443
step 226300: train 3.5775 | val 3.5498
step 226400: train 3.3693 | val 3.3256
step 226500: train 3.4278 | val 3.3658
step 226600: train 3.5095 | val 3.3959
step 226700: train 3.3267 | val 3.4465
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38799.48it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8217.80it/s]
step 226800: train 3.2974 | val 3.3576
step 226900: train 3.4537 | val 3.4272
step 227000: train 3.2369 | val 3.5542
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 227100: train 3.3390 | val 3.4495
step 227200: train 3.6046 | val 3.4534
step 227300: train 3.0580 | val 3.4886
step 227400: train 3.4464 | val 3.5528
step 227500: train 3.4022 | val 3.3284
step 227600: train 3.2653 | val 3.4744
step 227700: train 3.1794 | val 3.4277
step 227800: train 3.3174 | val 3.4300
step 227900: train 3.3048 | val 3.3786
step 228000: train 3.2622 | val 3.3866
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26233.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12533.41it/s]
step 228100: train 3.3521 | val 3.4467
step 228200: train 3.3878 | val 3.3996
step 228300: train 3.2993 | val 3.4285
step 228400: train 3.1413 | val 3.6463
step 228500: train 3.4446 | val 3.4420
step 228600: train 3.1375 | val 3.4503
step 228700: train 3.3203 | val 3.5028
step 228800: train 3.3387 | val 3.4756
step 228900: train 3.1174 | val 3.3882
step 229000: train 3.1194 | val 3.5477
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 229100: train 3.2588 | val 3.3487
step 229200: train 3.3692 | val 3.3930
step 229300: train 3.3839 | val 3.4220
step 229400: train 3.6527 | val 3.4598
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39394.49it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8113.56it/s]
step 229500: train 3.3893 | val 3.3561
step 229600: train 3.3945 | val 3.4100
step 229700: train 3.2237 | val 3.5475
step 229800: train 3.2435 | val 3.4307
step 229900: train 3.4863 | val 3.4465
step 230000: train 2.8630 | val 3.4181
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 230100: train 3.4543 | val 3.5269
step 230200: train 3.5203 | val 3.3081
step 230300: train 3.6300 | val 3.4448
step 230400: train 3.4757 | val 3.3966
step 230500: train 3.4901 | val 3.3752
step 230600: train 3.2708 | val 3.3452
step 230700: train 3.5078 | val 3.3594
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28947.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8435.61it/s]
step 230800: train 3.3617 | val 3.3884
step 230900: train 3.5117 | val 3.3432
step 231000: train 3.5382 | val 3.3754
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 231100: train 3.4038 | val 3.5772
step 231200: train 3.4688 | val 3.4180
step 231300: train 3.5079 | val 3.4035
step 231400: train 3.3898 | val 3.4328
step 231500: train 3.3725 | val 3.4215
step 231600: train 3.3812 | val 3.3327
step 231700: train 3.4104 | val 3.5170
step 231800: train 3.4423 | val 3.3118
step 231900: train 3.3923 | val 3.3416
step 232000: train 3.4964 | val 3.3888
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 232100: train 3.4277 | val 3.4249
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25892.49it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8388.01it/s]
step 232200: train 3.5523 | val 3.3321
step 232300: train 3.3367 | val 3.4019
step 232400: train 3.3084 | val 3.5352
step 232500: train 3.4001 | val 3.4259
step 232600: train 3.6038 | val 3.4404
step 232700: train 3.4319 | val 3.4483
step 232800: train 3.3773 | val 3.5361
step 232900: train 3.5885 | val 3.2969
step 233000: train 3.5311 | val 3.4577
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 233100: train 3.2740 | val 3.3885
step 233200: train 3.3616 | val 3.3977
step 233300: train 3.3752 | val 3.3597
step 233400: train 3.4220 | val 3.3596
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27728.39it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 6288.38it/s]
step 233500: train 3.4626 | val 3.4196
step 233600: train 3.2274 | val 3.3711
step 233700: train 3.4889 | val 3.3946
step 233800: train 3.2823 | val 3.6238
step 233900: train 3.3008 | val 3.4225
step 234000: train 3.6134 | val 3.4214
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 234100: train 3.1584 | val 3.4644
step 234200: train 3.5762 | val 3.4469
step 234300: train 3.2808 | val 3.3535
step 234400: train 3.4464 | val 3.5187
step 234500: train 3.2443 | val 3.3274
step 234600: train 3.4045 | val 3.3750
step 234700: train 3.3309 | val 3.4027
step 234800: train 3.2635 | val 3.4468
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31364.59it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17459.12it/s]
step 234900: train 3.4326 | val 3.3550
step 235000: train 3.3232 | val 3.4221
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 235100: train 3.4575 | val 3.5533
step 235200: train 3.2194 | val 3.4551
step 235300: train 3.3630 | val 3.4703
step 235400: train 3.2151 | val 3.4870
step 235500: train 3.1976 | val 3.5628
step 235600: train 3.1038 | val 3.3378
step 235700: train 3.4942 | val 3.4821
step 235800: train 3.2059 | val 3.4103
step 235900: train 3.1503 | val 3.4211
step 236000: train 3.2724 | val 3.3839
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 236100: train 3.4621 | val 3.3869
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26435.84it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9203.66it/s]
step 236200: train 3.5630 | val 3.4473
step 236300: train 3.1810 | val 3.4003
step 236400: train 3.3776 | val 3.4246
step 236500: train 3.2386 | val 3.6413
step 236600: train 3.3224 | val 3.4395
step 236700: train 3.5741 | val 3.4415
step 236800: train 3.1122 | val 3.4873
step 236900: train 3.4694 | val 3.4642
step 237000: train 3.4884 | val 3.3530
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 237100: train 3.5283 | val 3.5156
step 237200: train 3.3472 | val 3.3094
step 237300: train 3.4204 | val 3.3524
step 237400: train 3.4159 | val 3.3947
step 237500: train 3.4474 | val 3.4338
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44484.76it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8525.75it/s]
step 237600: train 3.6054 | val 3.3363
step 237700: train 3.3711 | val 3.4012
step 237800: train 3.5014 | val 3.5249
step 237900: train 3.2715 | val 3.4253
step 238000: train 3.5786 | val 3.4366
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 238100: train 3.4500 | val 3.4516
step 238200: train 3.3424 | val 3.5129
step 238300: train 3.3773 | val 3.3003
step 238400: train 3.3095 | val 3.4413
step 238500: train 3.5539 | val 3.3931
step 238600: train 3.4266 | val 3.3698
step 238700: train 3.3675 | val 3.3472
step 238800: train 3.4294 | val 3.3476
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30255.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8801.39it/s]
step 238900: train 3.5243 | val 3.4071
step 239000: train 3.3121 | val 3.3567
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 239100: train 3.4382 | val 3.3977
step 239200: train 3.7022 | val 3.5963
step 239300: train 3.5752 | val 3.4125
step 239400: train 3.3852 | val 3.4043
step 239500: train 3.4974 | val 3.4632
step 239600: train 3.3451 | val 3.4254
step 239700: train 3.2836 | val 3.3223
step 239800: train 3.3952 | val 3.5151
step 239900: train 3.3277 | val 3.3096
step 240000: train 3.4942 | val 3.3520
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 240100: train 3.4309 | val 3.3906
step 240200: train 3.3284 | val 3.4311
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27807.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8520.55it/s]
step 240300: train 3.3893 | val 3.3466
step 240400: train 3.3338 | val 3.3967
step 240500: train 3.4342 | val 3.5297
step 240600: train 3.5743 | val 3.4328
step 240700: train 3.3361 | val 3.4391
step 240800: train 3.3953 | val 3.4678
step 240900: train 3.4220 | val 3.5409
step 241000: train 3.5855 | val 3.3183
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 241100: train 3.4603 | val 3.4517
step 241200: train 3.4685 | val 3.4022
step 241300: train 3.2396 | val 3.3979
step 241400: train 3.0818 | val 3.3715
step 241500: train 3.3786 | val 3.3630
Resolving data files: 100%
 2410/2410 [00:00<00:00, 33929.14it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8506.61it/s]
step 241600: train 3.4749 | val 3.4233
step 241700: train 3.3031 | val 3.3767
step 241800: train 3.4714 | val 3.4004
step 241900: train 3.2576 | val 3.6266
step 242000: train 3.3500 | val 3.4260
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 242100: train 3.4009 | val 3.4344
step 242200: train 3.3842 | val 3.4824
step 242300: train 3.2657 | val 3.4522
step 242400: train 3.2693 | val 3.3600
step 242500: train 3.4451 | val 3.5264
step 242600: train 3.1821 | val 3.3314
step 242700: train 3.1035 | val 3.3747
step 242800: train 3.3428 | val 3.4076
step 242900: train 3.3616 | val 3.4661
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44502.58it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16071.89it/s]
step 243000: train 3.2486 | val 3.3681
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 243100: train 3.2745 | val 3.4380
step 243200: train 3.2309 | val 3.5581
step 243300: train 3.2668 | val 3.4687
step 243400: train 3.2021 | val 3.4713
step 243500: train 3.6231 | val 3.4923
step 243600: train 3.2292 | val 3.5780
step 243700: train 3.2616 | val 3.3398
step 243800: train 3.2301 | val 3.4770
step 243900: train 3.2538 | val 3.4279
step 244000: train 3.3131 | val 3.4155
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 244100: train 3.2638 | val 3.3837
step 244200: train 3.5241 | val 3.3778
Resolving data files: 100%
 2410/2410 [00:00<00:00, 46180.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17608.86it/s]
step 244300: train 3.2495 | val 3.4647
step 244400: train 3.4889 | val 3.3908
step 244500: train 3.1211 | val 3.4088
step 244600: train 3.3459 | val 3.6065
step 244700: train 3.2384 | val 3.4151
step 244800: train 3.4290 | val 3.4081
step 244900: train 3.3358 | val 3.4621
step 245000: train 3.4624 | val 3.4241
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 245100: train 3.4749 | val 3.3344
step 245200: train 3.4335 | val 3.5081
step 245300: train 3.4653 | val 3.3080
step 245400: train 3.5670 | val 3.3405
step 245500: train 3.2225 | val 3.3820
step 245600: train 3.5782 | val 3.4116
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39835.56it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8468.21it/s]
step 245700: train 3.3957 | val 3.3294
step 245800: train 3.6463 | val 3.3916
step 245900: train 3.6249 | val 3.5203
step 246000: train 3.2968 | val 3.4193
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 246100: train 3.3023 | val 3.4240
step 246200: train 3.4521 | val 3.4432
step 246300: train 3.5220 | val 3.5132
step 246400: train 3.4341 | val 3.2923
step 246500: train 3.2812 | val 3.4474
step 246600: train 3.5670 | val 3.3935
step 246700: train 3.3893 | val 3.3626
step 246800: train 3.3135 | val 3.3475
step 246900: train 3.3532 | val 3.3459
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27551.24it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9030.41it/s]
step 247000: train 3.5118 | val 3.4064
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 247100: train 3.6536 | val 3.3564
step 247200: train 3.1899 | val 3.3758
step 247300: train 3.3417 | val 3.5941
step 247400: train 3.3834 | val 3.4084
step 247500: train 3.5086 | val 3.4068
step 247600: train 3.2106 | val 3.4532
step 247700: train 3.4505 | val 3.4210
step 247800: train 3.4898 | val 3.3273
step 247900: train 3.5764 | val 3.5060
step 248000: train 3.3603 | val 3.2999
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 248100: train 3.3582 | val 3.3446
step 248200: train 3.4730 | val 3.3816
step 248300: train 3.3552 | val 3.4329
Resolving data files: 100%
 2410/2410 [00:00<00:00, 29845.82it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12034.81it/s]
step 248400: train 3.2907 | val 3.3472
step 248500: train 3.2833 | val 3.4035
step 248600: train 3.6715 | val 3.5384
step 248700: train 3.4564 | val 3.4389
step 248800: train 3.5302 | val 3.4502
step 248900: train 3.4668 | val 3.4607
step 249000: train 3.2738 | val 3.5356
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 249100: train 3.4264 | val 3.3117
step 249200: train 3.1992 | val 3.4664
step 249300: train 3.4844 | val 3.3921
step 249400: train 3.1467 | val 3.3935
step 249500: train 3.3310 | val 3.3644
step 249600: train 3.3425 | val 3.3563
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27716.53it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8308.14it/s]
step 249700: train 3.3645 | val 3.4184
step 249800: train 3.3506 | val 3.3769
step 249900: train 3.3134 | val 3.3960
step 250000: train 3.2549 | val 3.6298
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 250100: train 3.1143 | val 3.4287
step 250200: train 3.2149 | val 3.4351
step 250300: train 3.3532 | val 3.4827
step 250400: train 3.2400 | val 3.4588
step 250500: train 3.4176 | val 3.3687
step 250600: train 3.1859 | val 3.5385
step 250700: train 3.0567 | val 3.3396
step 250800: train 3.0554 | val 3.3839
step 250900: train 3.3398 | val 3.4095
step 251000: train 3.2194 | val 3.4630
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25328.49it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8666.05it/s]
step 251100: train 3.1674 | val 3.3700
step 251200: train 3.3986 | val 3.4396
step 251300: train 3.2146 | val 3.5703
step 251400: train 3.0704 | val 3.4829
step 251500: train 3.1516 | val 3.4781
step 251600: train 3.3858 | val 3.4889
step 251700: train 3.3570 | val 3.5654
step 251800: train 3.4705 | val 3.3351
step 251900: train 3.4604 | val 3.4559
step 252000: train 3.5467 | val 3.3942
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 252100: train 3.3388 | val 3.3775
step 252200: train 3.6138 | val 3.3426
step 252300: train 3.3463 | val 3.3426
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32750.27it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9460.48it/s]
step 252400: train 3.4171 | val 3.3986
step 252500: train 3.2628 | val 3.3479
step 252600: train 3.4868 | val 3.3646
step 252700: train 3.3160 | val 3.5866
step 252800: train 3.4926 | val 3.3997
step 252900: train 3.3851 | val 3.3947
step 253000: train 3.4786 | val 3.4504
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 253100: train 3.1427 | val 3.4089
step 253200: train 3.5078 | val 3.3232
step 253300: train 3.4276 | val 3.4921
step 253400: train 3.2870 | val 3.3024
step 253500: train 3.4088 | val 3.3245
step 253600: train 3.4219 | val 3.3711
step 253700: train 3.1821 | val 3.3994
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41809.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8965.61it/s]
step 253800: train 3.2853 | val 3.3158
step 253900: train 3.3842 | val 3.3690
step 254000: train 3.4078 | val 3.5055
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 254100: train 3.4768 | val 3.3988
step 254200: train 3.3676 | val 3.4207
step 254300: train 3.5187 | val 3.4469
step 254400: train 3.3065 | val 3.5110
step 254500: train 3.4102 | val 3.2813
step 254600: train 3.2825 | val 3.4272
step 254700: train 3.4588 | val 3.3742
step 254800: train 3.1515 | val 3.3656
step 254900: train 3.3212 | val 3.3350
step 255000: train 3.1517 | val 3.3446
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42053.32it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8681.93it/s]
step 255100: train 3.3493 | val 3.3985
step 255200: train 3.3862 | val 3.3569
step 255300: train 3.2667 | val 3.3752
step 255400: train 3.4034 | val 3.5916
step 255500: train 3.3814 | val 3.3903
step 255600: train 3.3529 | val 3.3939
step 255700: train 3.2113 | val 3.4524
step 255800: train 3.3079 | val 3.4156
step 255900: train 3.4068 | val 3.3132
step 256000: train 3.4184 | val 3.4979
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 256100: train 3.5220 | val 3.2990
step 256200: train 3.3984 | val 3.3442
step 256300: train 3.4564 | val 3.3771
step 256400: train 3.3557 | val 3.4209
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26853.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8115.80it/s]
step 256500: train 3.5468 | val 3.3390
step 256600: train 3.3029 | val 3.3928
step 256700: train 3.3648 | val 3.5227
step 256800: train 3.3095 | val 3.4252
step 256900: train 3.4923 | val 3.4401
step 257000: train 3.4642 | val 3.4674
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 257100: train 3.3025 | val 3.5353
step 257200: train 3.2285 | val 3.3013
step 257300: train 3.2209 | val 3.4560
step 257400: train 3.2025 | val 3.3933
step 257500: train 3.3848 | val 3.3824
step 257600: train 3.3521 | val 3.3469
step 257700: train 3.4939 | val 3.3541
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25645.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8348.89it/s]
step 257800: train 3.4546 | val 3.4149
step 257900: train 3.1656 | val 3.3676
step 258000: train 3.3449 | val 3.3949
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 258100: train 3.3664 | val 3.6177
step 258200: train 3.3147 | val 3.4211
step 258300: train 3.4361 | val 3.4205
step 258400: train 3.5258 | val 3.4844
step 258500: train 3.2836 | val 3.4476
step 258600: train 3.2880 | val 3.3659
step 258700: train 3.3310 | val 3.5244
step 258800: train 3.4528 | val 3.3238
step 258900: train 3.4390 | val 3.3664
step 259000: train 3.3484 | val 3.4016
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 259100: train 3.1616 | val 3.4450
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26421.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10328.44it/s]
step 259200: train 3.5296 | val 3.3577
step 259300: train 3.3342 | val 3.4196
step 259400: train 3.2141 | val 3.5457
step 259500: train 3.0776 | val 3.4479
step 259600: train 3.0984 | val 3.4508
step 259700: train 3.0752 | val 3.4887
step 259800: train 2.9552 | val 3.5599
step 259900: train 3.5647 | val 3.3073
step 260000: train 3.2716 | val 3.4658
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 260100: train 3.4358 | val 3.3950
step 260200: train 3.3447 | val 3.3658
step 260300: train 3.4010 | val 3.3383
step 260400: train 3.3054 | val 3.3356
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45286.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8414.57it/s]
step 260500: train 3.3673 | val 3.3892
step 260600: train 3.3306 | val 3.3328
step 260700: train 3.3078 | val 3.3699
step 260800: train 3.5210 | val 3.5857
step 260900: train 3.7278 | val 3.3917
step 261000: train 3.2518 | val 3.3824
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 261100: train 3.1745 | val 3.4456
step 261200: train 3.3882 | val 3.4011
step 261300: train 3.3003 | val 3.3198
step 261400: train 3.4859 | val 3.4917
step 261500: train 3.4105 | val 3.2958
step 261600: train 3.3766 | val 3.3253
step 261700: train 3.2243 | val 3.3625
step 261800: train 3.7077 | val 3.4088
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26129.83it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8356.50it/s]
step 261900: train 3.4931 | val 3.3152
step 262000: train 3.4891 | val 3.3778
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 262100: train 3.5314 | val 3.5134
step 262200: train 3.3557 | val 3.4126
step 262300: train 3.2862 | val 3.4178
step 262400: train 3.4554 | val 3.4448
step 262500: train 3.3175 | val 3.5051
step 262600: train 3.3758 | val 3.2950
step 262700: train 3.5559 | val 3.4306
step 262800: train 3.3434 | val 3.3734
step 262900: train 3.3835 | val 3.3695
step 263000: train 3.2125 | val 3.3420
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 263100: train 3.4738 | val 3.3378
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28910.35it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8248.62it/s]
step 263200: train 3.3690 | val 3.3992
step 263300: train 3.4748 | val 3.3469
step 263400: train 3.3614 | val 3.3679
step 263500: train 3.5917 | val 3.5926
step 263600: train 3.5466 | val 3.3913
step 263700: train 3.2901 | val 3.4017
step 263800: train 3.3524 | val 3.4508
step 263900: train 3.1382 | val 3.4212
step 264000: train 3.2782 | val 3.3465
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 264100: train 3.3392 | val 3.5047
step 264200: train 3.3454 | val 3.3039
step 264300: train 3.3108 | val 3.3471
step 264400: train 3.2197 | val 3.3810
step 264500: train 3.5305 | val 3.4342
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24546.20it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8151.74it/s]
step 264600: train 3.3693 | val 3.3431
step 264700: train 3.2980 | val 3.3976
step 264800: train 3.2663 | val 3.5342
step 264900: train 3.4763 | val 3.4383
step 265000: train 3.3464 | val 3.4363
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 265100: train 3.5310 | val 3.4582
step 265200: train 3.3796 | val 3.5277
step 265300: train 3.1875 | val 3.2997
step 265400: train 3.1703 | val 3.4644
step 265500: train 3.4293 | val 3.4086
step 265600: train 3.3896 | val 3.3965
step 265700: train 3.2573 | val 3.3636
step 265800: train 3.2683 | val 3.3558
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26228.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7637.71it/s]
step 265900: train 3.3132 | val 3.4273
step 266000: train 3.1957 | val 3.3786
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 266100: train 3.0781 | val 3.3973
step 266200: train 3.3839 | val 3.6267
step 266300: train 3.2771 | val 3.4245
step 266400: train 3.2951 | val 3.4289
step 266500: train 3.2885 | val 3.4804
step 266600: train 3.1358 | val 3.4474
step 266700: train 3.2800 | val 3.3536
step 266800: train 3.2498 | val 3.5192
step 266900: train 3.2233 | val 3.3208
step 267000: train 3.4496 | val 3.3667
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 267100: train 3.1660 | val 3.3959
step 267200: train 3.1697 | val 3.4449
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24943.05it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8415.90it/s]
step 267300: train 3.4356 | val 3.3405
step 267400: train 3.2704 | val 3.3922
step 267500: train 3.3808 | val 3.5240
step 267600: train 3.4272 | val 3.4139
step 267700: train 3.4695 | val 3.4206
step 267800: train 3.3433 | val 3.4363
step 267900: train 3.4047 | val 3.5087
step 268000: train 3.3218 | val 3.2797
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 268100: train 3.1699 | val 3.4493
step 268200: train 3.4107 | val 3.3703
step 268300: train 3.4005 | val 3.3531
step 268400: train 3.3721 | val 3.3469
step 268500: train 3.5401 | val 3.3285
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25454.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8231.50it/s]
step 268600: train 3.4845 | val 3.3856
step 268700: train 3.4334 | val 3.3300
step 268800: train 3.4940 | val 3.3352
step 268900: train 3.0932 | val 3.5730
step 269000: train 3.4806 | val 3.3776
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 269100: train 3.2145 | val 3.3754
step 269200: train 3.2935 | val 3.4286
step 269300: train 3.3302 | val 3.3881
step 269400: train 3.2236 | val 3.3054
step 269500: train 3.4583 | val 3.4922
step 269600: train 3.4796 | val 3.2826
step 269700: train 3.5100 | val 3.3240
step 269800: train 3.3399 | val 3.3702
step 269900: train 3.4690 | val 3.4039
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27494.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8919.58it/s]
step 270000: train 3.4636 | val 3.3163
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 270100: train 3.3473 | val 3.3637
step 270200: train 3.3350 | val 3.4990
step 270300: train 3.3880 | val 3.3984
step 270400: train 3.3038 | val 3.4087
step 270500: train 3.4131 | val 3.4528
step 270600: train 3.4106 | val 3.4898
step 270700: train 3.3361 | val 3.2805
step 270800: train 3.3451 | val 3.4211
step 270900: train 3.3628 | val 3.3748
step 271000: train 3.2938 | val 3.3614
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 271100: train 3.4854 | val 3.3357
step 271200: train 3.6070 | val 3.3430
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24178.19it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15201.87it/s]
step 271300: train 3.3569 | val 3.4034
step 271400: train 3.3886 | val 3.3475
step 271500: train 3.3666 | val 3.3678
step 271600: train 3.5147 | val 3.5929
step 271700: train 3.4084 | val 3.3958
step 271800: train 3.2424 | val 3.4051
step 271900: train 3.4358 | val 3.4486
step 272000: train 3.4344 | val 3.4152
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 272100: train 3.3443 | val 3.3517
step 272200: train 3.3326 | val 3.5240
step 272300: train 3.2902 | val 3.3002
step 272400: train 3.4126 | val 3.3419
step 272500: train 3.3372 | val 3.3704
step 272600: train 3.4619 | val 3.4178
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43826.68it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8508.21it/s]
step 272700: train 3.3137 | val 3.3338
step 272800: train 3.0902 | val 3.3977
step 272900: train 3.1282 | val 3.5332
step 273000: train 3.4160 | val 3.4469
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 273100: train 3.3429 | val 3.4532
step 273200: train 3.0123 | val 3.4643
step 273300: train 3.1229 | val 3.5409
step 273400: train 3.1785 | val 3.3177
step 273500: train 3.2141 | val 3.4567
step 273600: train 3.4921 | val 3.4080
step 273700: train 3.3270 | val 3.3946
step 273800: train 3.0427 | val 3.3680
step 273900: train 3.1591 | val 3.3709
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28521.33it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8443.61it/s]
step 274000: train 3.1564 | val 3.4261
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 274100: train 3.0903 | val 3.3732
step 274200: train 3.3349 | val 3.3995
step 274300: train 3.1372 | val 3.6383
step 274400: train 3.1101 | val 3.4170
step 274500: train 3.3385 | val 3.4298
step 274600: train 3.3225 | val 3.4746
step 274700: train 3.3843 | val 3.4234
step 274800: train 3.3806 | val 3.3302
step 274900: train 3.4358 | val 3.5023
step 275000: train 3.3233 | val 3.2878
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 275100: train 3.3684 | val 3.3228
step 275200: train 3.4373 | val 3.3563
step 275300: train 3.2966 | val 3.4008
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25157.03it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8368.88it/s]
step 275400: train 3.4139 | val 3.3033
step 275500: train 3.2718 | val 3.3666
step 275600: train 3.3747 | val 3.4994
step 275700: train 3.3371 | val 3.3969
step 275800: train 3.2893 | val 3.3997
step 275900: train 3.3547 | val 3.4252
step 276000: train 3.4825 | val 3.4805
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 276100: train 3.5891 | val 3.2654
step 276200: train 3.3441 | val 3.4101
step 276300: train 3.1303 | val 3.3611
step 276400: train 3.2360 | val 3.3458
step 276500: train 3.3444 | val 3.3200
step 276600: train 3.2459 | val 3.3178
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25527.55it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15571.53it/s]
step 276700: train 3.4549 | val 3.3708
step 276800: train 3.3109 | val 3.3166
step 276900: train 3.4110 | val 3.3379
step 277000: train 3.4062 | val 3.5516
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 277100: train 3.5324 | val 3.3755
step 277200: train 3.4049 | val 3.3754
step 277300: train 3.1914 | val 3.4152
step 277400: train 3.4115 | val 3.3774
step 277500: train 3.6877 | val 3.2945
step 277600: train 3.4850 | val 3.4886
step 277700: train 3.4455 | val 3.2686
step 277800: train 3.2965 | val 3.3239
step 277900: train 3.6038 | val 3.3531
step 278000: train 3.3554 | val 3.3952
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25650.76it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8233.81it/s]
step 278100: train 3.3251 | val 3.3060
step 278200: train 3.4773 | val 3.3640
step 278300: train 3.3244 | val 3.4942
step 278400: train 3.2798 | val 3.3972
step 278500: train 3.4690 | val 3.4044
step 278600: train 3.4590 | val 3.4379
step 278700: train 3.3856 | val 3.4876
step 278800: train 3.2657 | val 3.2711
step 278900: train 3.4953 | val 3.4232
step 279000: train 3.3271 | val 3.3683
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 279100: train 3.4616 | val 3.3516
step 279200: train 3.2992 | val 3.3303
step 279300: train 3.2434 | val 3.3208
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25703.79it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8707.81it/s]
step 279400: train 3.2775 | val 3.3912
step 279500: train 3.2258 | val 3.3402
step 279600: train 3.4417 | val 3.3678
step 279700: train 3.3657 | val 3.5819
step 279800: train 3.3071 | val 3.3887
step 279900: train 3.3008 | val 3.3973
step 280000: train 3.5107 | val 3.4444
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 280100: train 3.7279 | val 3.4018
step 280200: train 3.3310 | val 3.3153
step 280300: train 3.3576 | val 3.4961
step 280400: train 3.4255 | val 3.2918
step 280500: train 2.6869 | val 3.3352
step 280600: train 3.2354 | val 3.3670
step 280700: train 3.4227 | val 3.4181
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26006.80it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7945.15it/s]
step 280800: train 3.3457 | val 3.3253
step 280900: train 3.6792 | val 3.3995
step 281000: train 3.3385 | val 3.5264
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 281100: train 3.4406 | val 3.4266
step 281200: train 3.3327 | val 3.4384
step 281300: train 3.2928 | val 3.4606
step 281400: train 3.2637 | val 3.5292
step 281500: train 3.0779 | val 3.3094
step 281600: train 3.2431 | val 3.4556
step 281700: train 3.2696 | val 3.3882
step 281800: train 3.2876 | val 3.3879
step 281900: train 3.1277 | val 3.3479
step 282000: train 3.2121 | val 3.3461
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31777.73it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7950.32it/s]
step 282100: train 3.1997 | val 3.4185
step 282200: train 3.2957 | val 3.3645
step 282300: train 3.2369 | val 3.3900
step 282400: train 3.1624 | val 3.6150
step 282500: train 3.2163 | val 3.4054
step 282600: train 2.9484 | val 3.4143
step 282700: train 3.2844 | val 3.4465
step 282800: train 3.3086 | val 3.4085
step 282900: train 3.4526 | val 3.2983
step 283000: train 3.3083 | val 3.4826
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 283100: train 3.3185 | val 3.2749
step 283200: train 3.1462 | val 3.3125
step 283300: train 3.3020 | val 3.3468
step 283400: train 3.3471 | val 3.3823
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42096.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8346.16it/s]
step 283500: train 3.1967 | val 3.2951
step 283600: train 3.4034 | val 3.3495
step 283700: train 3.2746 | val 3.4875
step 283800: train 3.3216 | val 3.3767
step 283900: train 3.4119 | val 3.4042
step 284000: train 3.3390 | val 3.4109
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 284100: train 3.4496 | val 3.4758
step 284200: train 3.3566 | val 3.2504
step 284300: train 3.5017 | val 3.4132
step 284400: train 3.1657 | val 3.3434
step 284500: train 3.3678 | val 3.3358
step 284600: train 3.4661 | val 3.3082
step 284700: train 3.3548 | val 3.3103
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25081.94it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8467.84it/s]
step 284800: train 3.4074 | val 3.3718
step 284900: train 3.3666 | val 3.3075
step 285000: train 3.3856 | val 3.3445
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 285100: train 3.3980 | val 3.5620
step 285200: train 3.2893 | val 3.3740
step 285300: train 3.5109 | val 3.3720
step 285400: train 3.4673 | val 3.4163
step 285500: train 3.3843 | val 3.3789
step 285600: train 3.0885 | val 3.2915
step 285700: train 3.5379 | val 3.4821
step 285800: train 3.3380 | val 3.2764
step 285900: train 3.6202 | val 3.3101
step 286000: train 3.2117 | val 3.3440
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 286100: train 3.4285 | val 3.3851
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30892.03it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8475.91it/s]
step 286200: train 3.7970 | val 3.3066
step 286300: train 3.3155 | val 3.3624
step 286400: train 3.2758 | val 3.4965
step 286500: train 3.3449 | val 3.4007
step 286600: train 3.3584 | val 3.4067
step 286700: train 3.2962 | val 3.4286
step 286800: train 3.4102 | val 3.5070
step 286900: train 3.1496 | val 3.2773
step 287000: train 3.2676 | val 3.4232
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 287100: train 3.4074 | val 3.3674
step 287200: train 3.2993 | val 3.3621
step 287300: train 3.2351 | val 3.3223
step 287400: train 3.3420 | val 3.3297
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25984.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12677.63it/s]
step 287500: train 3.1635 | val 3.3988
step 287600: train 3.2133 | val 3.3400
step 287700: train 3.2910 | val 3.3726
step 287800: train 3.2577 | val 3.5907
step 287900: train 3.2965 | val 3.3840
step 288000: train 3.6167 | val 3.3856
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 288100: train 3.4146 | val 3.4396
step 288200: train 3.4308 | val 3.4139
step 288300: train 3.3760 | val 3.3349
step 288400: train 3.1810 | val 3.4884
step 288500: train 3.1708 | val 3.3078
step 288600: train 3.4827 | val 3.3320
step 288700: train 3.4678 | val 3.3708
step 288800: train 3.2528 | val 3.4223
Resolving data files: 100%
 2410/2410 [00:00<00:00, 29247.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13862.85it/s]
step 288900: train 3.2183 | val 3.3339
step 289000: train 3.3346 | val 3.3993
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 289100: train 3.1819 | val 3.5276
step 289200: train 3.2251 | val 3.4226
step 289300: train 3.7487 | val 3.4352
step 289400: train 3.2895 | val 3.4480
step 289500: train 3.3997 | val 3.5298
step 289600: train 3.4629 | val 3.3050
step 289700: train 3.3523 | val 3.4477
step 289800: train 3.2012 | val 3.3854
step 289900: train 3.1898 | val 3.3812
step 290000: train 3.2063 | val 3.3443
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 290100: train 3.4854 | val 3.3385
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32482.95it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12272.76it/s]
step 290200: train 3.4550 | val 3.3947
step 290300: train 3.3312 | val 3.3314
step 290400: train 3.4426 | val 3.3537
step 290500: train 3.2837 | val 3.5882
step 290600: train 3.4373 | val 3.3707
step 290700: train 3.4080 | val 3.3725
step 290800: train 3.4150 | val 3.4216
step 290900: train 3.3824 | val 3.3804
step 291000: train 3.3755 | val 3.2881
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 291100: train 3.4682 | val 3.4727
step 291200: train 3.3465 | val 3.2680
step 291300: train 3.4546 | val 3.3058
step 291400: train 3.2653 | val 3.3360
step 291500: train 3.4240 | val 3.3742
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26624.82it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8756.38it/s]
step 291600: train 3.5050 | val 3.2945
step 291700: train 3.2921 | val 3.3499
step 291800: train 3.3890 | val 3.4828
step 291900: train 3.2942 | val 3.3771
step 292000: train 3.3338 | val 3.3897
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 292100: train 3.3745 | val 3.4144
step 292200: train 3.3162 | val 3.4794
step 292300: train 3.5118 | val 3.2559
step 292400: train 3.2485 | val 3.4048
step 292500: train 3.3497 | val 3.3493
step 292600: train 3.2340 | val 3.3399
step 292700: train 3.3531 | val 3.3115
step 292800: train 3.3166 | val 3.3101
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24602.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8337.27it/s]
step 292900: train 3.3743 | val 3.3705
step 293000: train 3.3092 | val 3.3140
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 293100: train 3.4757 | val 3.3538
step 293200: train 3.4279 | val 3.5620
step 293300: train 3.4371 | val 3.3728
step 293400: train 3.2869 | val 3.3708
step 293500: train 3.4064 | val 3.4165
step 293600: train 3.4679 | val 3.3819
step 293700: train 3.3951 | val 3.2932
step 293800: train 3.2027 | val 3.4716
step 293900: train 3.2929 | val 3.2735
step 294000: train 3.3834 | val 3.3183
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 294100: train 3.1825 | val 3.3491
step 294200: train 3.4919 | val 3.3966
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26384.50it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8631.65it/s]
step 294300: train 3.4696 | val 3.3077
step 294400: train 3.2422 | val 3.3646
step 294500: train 3.4248 | val 3.5128
step 294600: train 3.2814 | val 3.3947
step 294700: train 3.2618 | val 3.4044
step 294800: train 3.5155 | val 3.4311
step 294900: train 3.3683 | val 3.5078
step 295000: train 3.4469 | val 3.2876
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 295100: train 3.3306 | val 3.4609
step 295200: train 3.2933 | val 3.3596
step 295300: train 3.3525 | val 3.3688
step 295400: train 3.3567 | val 3.3251
step 295500: train 3.3581 | val 3.3195
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24217.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15216.44it/s]
step 295600: train 3.3743 | val 3.3915
step 295700: train 3.1105 | val 3.3468
step 295800: train 3.1833 | val 3.3729
step 295900: train 3.1487 | val 3.5887
step 296000: train 3.2230 | val 3.3980
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 296100: train 3.0839 | val 3.3983
step 296200: train 3.2125 | val 3.4534
step 296300: train 3.0617 | val 3.4293
step 296400: train 3.3236 | val 3.3287
step 296500: train 3.2750 | val 3.5084
step 296600: train 3.2358 | val 3.3062
step 296700: train 3.0461 | val 3.3531
step 296800: train 3.1333 | val 3.3769
step 296900: train 3.3383 | val 3.4335
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26035.21it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16271.86it/s]
step 297000: train 3.2221 | val 3.3379
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 297100: train 3.2232 | val 3.4028
step 297200: train 3.3090 | val 3.5307
step 297300: train 3.3956 | val 3.4265
step 297400: train 3.1555 | val 3.4365
step 297500: train 3.7074 | val 3.4456
step 297600: train 3.4225 | val 3.5016
step 297700: train 3.6099 | val 3.2812
step 297800: train 3.3567 | val 3.4108
step 297900: train 3.4099 | val 3.3546
step 298000: train 3.5318 | val 3.3433
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 298100: train 3.6964 | val 3.3152
step 298200: train 3.3984 | val 3.3009
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26145.31it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7940.75it/s]
step 298300: train 3.4046 | val 3.3652
step 298400: train 3.3396 | val 3.3192
step 298500: train 3.4210 | val 3.3415
step 298600: train 3.4979 | val 3.5436
step 298700: train 3.3288 | val 3.3567
step 298800: train 3.3895 | val 3.3644
step 298900: train 4.0881 | val 3.4100
step 299000: train 3.3760 | val 3.3733
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 299100: train 3.4099 | val 3.2873
step 299200: train 3.5443 | val 3.4677
step 299300: train 3.2925 | val 3.2564
step 299400: train 3.2523 | val 3.2994
step 299500: train 3.5049 | val 3.3293
step 299600: train 3.3486 | val 3.3720
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42734.24it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9059.11it/s]
step 299700: train 3.4414 | val 3.2881
step 299800: train 3.4864 | val 3.3370
step 299900: train 3.4086 | val 3.4781
step 300000: train 3.4113 | val 3.3700
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 300100: train 3.4884 | val 3.3808
step 300200: train 3.3384 | val 3.3982
step 300300: train 3.4751 | val 3.4699
step 300400: train 3.3086 | val 3.2467
step 300500: train 3.3236 | val 3.3979
step 300600: train 3.5332 | val 3.3453
step 300700: train 3.1952 | val 3.3306
step 300800: train 3.2893 | val 3.2985
step 300900: train 3.4920 | val 3.3029
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25455.30it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8791.38it/s]
step 301000: train 3.4231 | val 3.3609
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 301100: train 3.5392 | val 3.2887
step 301200: train 3.2749 | val 3.3358
step 301300: train 3.2161 | val 3.5505
step 301400: train 3.4605 | val 3.3562
step 301500: train 3.3653 | val 3.3612
step 301600: train 3.3024 | val 3.4144
step 301700: train 3.4716 | val 3.3734
step 301800: train 3.4492 | val 3.2853
step 301900: train 3.1713 | val 3.4607
step 302000: train 3.3819 | val 3.2647
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 302100: train 3.3547 | val 3.3136
step 302200: train 3.3044 | val 3.3400
step 302300: train 3.1513 | val 3.3895
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25488.03it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14445.33it/s]
step 302400: train 3.4147 | val 3.3002
step 302500: train 3.5555 | val 3.3526
step 302600: train 3.2781 | val 3.4984
step 302700: train 3.4049 | val 3.4002
step 302800: train 3.3283 | val 3.3963
step 302900: train 3.4058 | val 3.4179
step 303000: train 3.3363 | val 3.5000
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 303100: train 3.2921 | val 3.2680
step 303200: train 3.2461 | val 3.4131
step 303300: train 3.0542 | val 3.3551
step 303400: train 3.2394 | val 3.3551
step 303500: train 3.5021 | val 3.3175
step 303600: train 3.1153 | val 3.3198
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43236.73it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16409.64it/s]
step 303700: train 3.3246 | val 3.3871
step 303800: train 3.4110 | val 3.3288
step 303900: train 3.2919 | val 3.3643
step 304000: train 3.2029 | val 3.5774
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 304100: train 3.2237 | val 3.3827
step 304200: train 3.1639 | val 3.3880
step 304300: train 3.1604 | val 3.4391
step 304400: train 3.2580 | val 3.4123
step 304500: train 3.1409 | val 3.3181
step 304600: train 3.1679 | val 3.4820
step 304700: train 3.4005 | val 3.2850
step 304800: train 3.1795 | val 3.3391
step 304900: train 3.2044 | val 3.3662
step 305000: train 3.2468 | val 3.4151
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28361.04it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14260.80it/s]
step 305100: train 3.1993 | val 3.3339
step 305200: train 3.4444 | val 3.3837
step 305300: train 3.2291 | val 3.5165
step 305400: train 3.2032 | val 3.4133
step 305500: train 3.3468 | val 3.4326
step 305600: train 3.3576 | val 3.4417
step 305700: train 3.3945 | val 3.5084
step 305800: train 3.3373 | val 3.2589
step 305900: train 3.3371 | val 3.3969
step 306000: train 3.3358 | val 3.3421
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 306100: train 3.3937 | val 3.3339
step 306200: train 3.4353 | val 3.2971
step 306300: train 3.3182 | val 3.2996
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30031.86it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9100.25it/s]
step 306400: train 3.4422 | val 3.3528
step 306500: train 3.1461 | val 3.2920
step 306600: train 3.4353 | val 3.3183
step 306700: train 3.4078 | val 3.5403
step 306800: train 3.2309 | val 3.3464
step 306900: train 3.3681 | val 3.3524
step 307000: train 3.1754 | val 3.3926
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 307100: train 3.3924 | val 3.3587
step 307200: train 3.4345 | val 3.2640
step 307300: train 3.4693 | val 3.4519
step 307400: train 3.1864 | val 3.2490
step 307500: train 3.3841 | val 3.2839
step 307600: train 3.3501 | val 3.3348
step 307700: train 3.3495 | val 3.3609
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44422.20it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8418.19it/s]
step 307800: train 3.3074 | val 3.2806
step 307900: train 3.2729 | val 3.3304
step 308000: train 3.5204 | val 3.4833
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 308100: train 3.4153 | val 3.3732
step 308200: train 3.5245 | val 3.3804
step 308300: train 3.4323 | val 3.4027
step 308400: train 3.5147 | val 3.4828
step 308500: train 3.3030 | val 3.2426
step 308600: train 3.3917 | val 3.3907
step 308700: train 3.7836 | val 3.3405
step 308800: train 3.2728 | val 3.3308
step 308900: train 3.3989 | val 3.3004
step 309000: train 3.3976 | val 3.2985
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38909.10it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 10169.95it/s]
step 309100: train 3.2308 | val 3.3602
step 309200: train 3.3652 | val 3.3012
step 309300: train 3.4122 | val 3.3358
step 309400: train 3.4077 | val 3.5601
step 309500: train 3.3362 | val 3.3562
step 309600: train 3.2251 | val 3.3609
step 309700: train 3.2109 | val 3.4158
step 309800: train 3.3194 | val 3.3827
step 309900: train 3.2125 | val 3.2992
step 310000: train 3.2767 | val 3.4570
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 310100: train 3.2584 | val 3.2615
step 310200: train 3.3266 | val 3.3126
step 310300: train 3.3441 | val 3.3430
step 310400: train 3.2448 | val 3.3886
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39805.91it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8093.99it/s]
step 310500: train 3.3461 | val 3.3025
step 310600: train 3.4026 | val 3.3598
step 310700: train 3.1716 | val 3.4932
step 310800: train 3.4238 | val 3.3895
step 310900: train 3.2096 | val 3.3960
step 311000: train 3.2302 | val 3.4186
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 311100: train 3.3978 | val 3.4921
step 311200: train 3.2149 | val 3.2646
step 311300: train 3.3176 | val 3.4182
step 311400: train 3.4203 | val 3.3524
step 311500: train 3.1087 | val 3.3493
step 311600: train 3.3630 | val 3.3242
step 311700: train 3.3860 | val 3.3198
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27241.32it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14723.87it/s]
step 311800: train 3.2560 | val 3.3871
step 311900: train 3.3352 | val 3.3335
step 312000: train 3.1512 | val 3.3667
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 312100: train 3.3079 | val 3.5853
step 312200: train 3.4769 | val 3.3898
step 312300: train 3.1893 | val 3.3885
step 312400: train 3.2441 | val 3.4395
step 312500: train 3.2492 | val 3.4094
step 312600: train 2.9937 | val 3.3136
step 312700: train 2.9910 | val 3.4811
step 312800: train 3.1422 | val 3.2879
step 312900: train 3.1820 | val 3.3274
step 313000: train 3.1718 | val 3.3547
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 313100: train 3.7499 | val 3.3940
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26277.37it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 20826.48it/s]
step 313200: train 3.3589 | val 3.2944
step 313300: train 3.5431 | val 3.3477
step 313400: train 3.3283 | val 3.4767
step 313500: train 3.4366 | val 3.3637
step 313600: train 3.4519 | val 3.3862
step 313700: train 3.4903 | val 3.3983
step 313800: train 3.3655 | val 3.4679
step 313900: train 3.2375 | val 3.2425
step 314000: train 3.4110 | val 3.3854
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 314100: train 3.2930 | val 3.3372
step 314200: train 3.4537 | val 3.3165
step 314300: train 3.3590 | val 3.2921
step 314400: train 3.4047 | val 3.2855
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25917.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8453.09it/s]
step 314500: train 3.3002 | val 3.3471
step 314600: train 3.2240 | val 3.2924
step 314700: train 3.3354 | val 3.3153
step 314800: train 3.3697 | val 3.5350
step 314900: train 3.3987 | val 3.3473
step 315000: train 3.3810 | val 3.3458
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 315100: train 3.4631 | val 3.3997
step 315200: train 3.4569 | val 3.3662
step 315300: train 3.1856 | val 3.2764
step 315400: train 3.5635 | val 3.4694
step 315500: train 3.2395 | val 3.2484
step 315600: train 3.2327 | val 3.2809
step 315700: train 3.2322 | val 3.3322
step 315800: train 3.1891 | val 3.3820
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40029.12it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9145.46it/s]
step 315900: train 3.4333 | val 3.2837
step 316000: train 3.4230 | val 3.3488
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 316100: train 3.2670 | val 3.4683
step 316200: train 3.4425 | val 3.3686
step 316300: train 3.1997 | val 3.3757
step 316400: train 3.3051 | val 3.4015
step 316500: train 3.8779 | val 3.4703
step 316600: train 3.4809 | val 3.2471
step 316700: train 3.3938 | val 3.3896
step 316800: train 3.1946 | val 3.3347
step 316900: train 3.4694 | val 3.3318
step 317000: train 3.3959 | val 3.2997
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 317100: train 3.4259 | val 3.2995
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25459.46it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8291.13it/s]
step 317200: train 3.4194 | val 3.3687
step 317300: train 3.3117 | val 3.3108
step 317400: train 3.2709 | val 3.3410
step 317500: train 3.0861 | val 3.5672
step 317600: train 3.1581 | val 3.3621
step 317700: train 3.2663 | val 3.3593
step 317800: train 3.2161 | val 3.4136
step 317900: train 3.3721 | val 3.3766
step 318000: train 3.3767 | val 3.2886
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 318100: train 3.2238 | val 3.4646
step 318200: train 3.2867 | val 3.2598
step 318300: train 3.3581 | val 3.2968
step 318400: train 3.1786 | val 3.3341
step 318500: train 3.2814 | val 3.3850
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26075.23it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8490.86it/s]
step 318600: train 3.1305 | val 3.3094
step 318700: train 3.1345 | val 3.3659
step 318800: train 3.2746 | val 3.5049
step 318900: train 3.0456 | val 3.4026
step 319000: train 3.1537 | val 3.4026
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 319100: train 2.9802 | val 3.4260
step 319200: train 3.3340 | val 3.5125
step 319300: train 3.2366 | val 3.2778
step 319400: train 3.0976 | val 3.4077
step 319500: train 3.2743 | val 3.3604
step 319600: train 3.2114 | val 3.3591
step 319700: train 3.0143 | val 3.3266
step 319800: train 3.3451 | val 3.3187
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44195.16it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16029.33it/s]
step 319900: train 3.1932 | val 3.3866
step 320000: train 3.3346 | val 3.3428
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 320100: train 2.9961 | val 3.3695
step 320200: train 3.0218 | val 3.5933
step 320300: train 3.1714 | val 3.3866
step 320400: train 3.3147 | val 3.3843
step 320500: train 3.1175 | val 3.4202
step 320600: train 3.3825 | val 3.3768
step 320700: train 3.2532 | val 3.2754
step 320800: train 3.4368 | val 3.4539
step 320900: train 3.2291 | val 3.2523
step 321000: train 3.2306 | val 3.2899
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 321100: train 3.2706 | val 3.3198
step 321200: train 3.3398 | val 3.3568
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39202.45it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9547.23it/s]
step 321300: train 3.4192 | val 3.2707
step 321400: train 3.2897 | val 3.3288
step 321500: train 3.2882 | val 3.4790
step 321600: train 3.3263 | val 3.3619
step 321700: train 3.3788 | val 3.3685
step 321800: train 3.2622 | val 3.3848
step 321900: train 3.5965 | val 3.4589
step 322000: train 3.2752 | val 3.2323
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 322100: train 3.1736 | val 3.3618
step 322200: train 3.5908 | val 3.3225
step 322300: train 3.3934 | val 3.3117
step 322400: train 3.4237 | val 3.2867
step 322500: train 3.3529 | val 3.2744
Resolving data files: 100%
 2410/2410 [00:00<00:00, 44580.31it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16842.18it/s]
step 322600: train 3.2257 | val 3.3385
step 322700: train 3.5813 | val 3.2838
step 322800: train 3.5344 | val 3.3133
step 322900: train 3.1705 | val 3.5256
step 323000: train 3.3758 | val 3.3390
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 323100: train 3.2011 | val 3.3390
step 323200: train 3.3095 | val 3.3965
step 323300: train 3.4113 | val 3.3543
step 323400: train 3.4102 | val 3.2622
step 323500: train 3.4095 | val 3.4432
step 323600: train 3.4664 | val 3.2422
step 323700: train 3.2950 | val 3.2829
step 323800: train 3.2742 | val 3.3222
step 323900: train 3.2446 | val 3.3712
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34691.15it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16035.02it/s]
step 324000: train 3.2976 | val 3.2673
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 324100: train 3.2712 | val 3.3264
step 324200: train 3.3797 | val 3.4606
step 324300: train 3.4791 | val 3.3594
step 324400: train 3.6120 | val 3.3685
step 324500: train 3.3932 | val 3.3873
step 324600: train 3.4801 | val 3.4610
step 324700: train 3.3586 | val 3.2334
step 324800: train 3.3480 | val 3.3738
step 324900: train 3.2219 | val 3.3224
step 325000: train 3.2580 | val 3.3191
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 325100: train 3.2827 | val 3.2905
step 325200: train 3.3224 | val 3.2918
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30743.76it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9630.22it/s]
step 325300: train 3.2261 | val 3.3543
step 325400: train 3.2724 | val 3.3013
step 325500: train 3.3118 | val 3.3353
step 325600: train 3.3395 | val 3.5458
step 325700: train 3.2456 | val 3.3481
step 325800: train 3.2261 | val 3.3504
step 325900: train 3.2376 | val 3.4050
step 326000: train 3.2687 | val 3.3697
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 326100: train 3.2244 | val 3.2838
step 326200: train 3.4173 | val 3.4544
step 326300: train 3.2597 | val 3.2540
step 326400: train 3.3874 | val 3.2979
step 326500: train 3.1682 | val 3.3263
step 326600: train 3.7849 | val 3.3812
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37584.91it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18023.41it/s]
step 326700: train 3.1848 | val 3.2845
step 326800: train 3.3544 | val 3.3588
step 326900: train 3.3484 | val 3.4919
step 327000: train 3.0406 | val 3.4026
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 327100: train 3.2635 | val 3.3906
step 327200: train 3.2533 | val 3.4134
step 327300: train 3.0325 | val 3.4895
step 327400: train 3.2190 | val 3.2593
step 327500: train 3.2719 | val 3.3986
step 327600: train 3.2187 | val 3.3554
step 327700: train 3.0702 | val 3.3434
step 327800: train 3.2479 | val 3.3122
step 327900: train 3.3537 | val 3.3105
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31200.20it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13152.71it/s]
step 328000: train 3.2047 | val 3.3848
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 328100: train 3.0531 | val 3.3187
step 328200: train 3.2544 | val 3.3471
step 328300: train 3.0533 | val 3.5717
step 328400: train 3.2905 | val 3.3716
step 328500: train 3.0662 | val 3.3755
step 328600: train 3.3097 | val 3.4187
step 328700: train 3.3566 | val 3.3715
step 328800: train 3.3550 | val 3.2570
step 328900: train 3.4494 | val 3.4409
step 329000: train 3.2392 | val 3.2418
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 329100: train 3.2714 | val 3.2816
step 329200: train 3.3621 | val 3.3212
step 329300: train 3.2618 | val 3.3516
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40753.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14068.78it/s]
step 329400: train 3.2878 | val 3.2564
step 329500: train 3.3245 | val 3.3150
step 329600: train 3.2169 | val 3.4610
step 329700: train 3.3849 | val 3.3444
step 329800: train 3.2021 | val 3.3597
step 329900: train 3.2437 | val 3.3784
step 330000: train 3.1559 | val 3.4407
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 330100: train 3.4819 | val 3.2069
step 330200: train 3.4116 | val 3.3540
step 330300: train 3.2825 | val 3.3151
step 330400: train 3.2843 | val 3.2919
step 330500: train 3.4526 | val 3.2744
step 330600: train 3.4190 | val 3.2760
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25831.35it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8151.85it/s]
step 330700: train 3.2908 | val 3.3423
step 330800: train 3.5304 | val 3.2765
step 330900: train 3.2394 | val 3.3060
step 331000: train 3.4344 | val 3.5380
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 331100: train 3.2300 | val 3.3284
step 331200: train 3.3325 | val 3.3259
step 331300: train 3.4428 | val 3.3820
step 331400: train 3.2752 | val 3.3453
step 331500: train 3.3778 | val 3.2497
step 331600: train 3.4098 | val 3.4315
step 331700: train 3.2889 | val 3.2358
step 331800: train 3.2367 | val 3.2728
step 331900: train 3.4526 | val 3.3235
step 332000: train 3.4097 | val 3.3594
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 30584.33it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17022.34it/s]
step 332100: train 3.3431 | val 3.2625
step 332200: train 3.5101 | val 3.3231
step 332300: train 3.3553 | val 3.4726
step 332400: train 3.2297 | val 3.3640
step 332500: train 3.4263 | val 3.3756
step 332600: train 3.0301 | val 3.3991
step 332700: train 3.4035 | val 3.4667
step 332800: train 3.2830 | val 3.2370
step 332900: train 3.3658 | val 3.3799
step 333000: train 3.2029 | val 3.3282
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 333100: train 3.1660 | val 3.3215
step 333200: train 3.3583 | val 3.2995
step 333300: train 3.3506 | val 3.2890
Resolving data files: 100%
 2410/2410 [00:00<00:00, 37093.63it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18591.77it/s]
step 333400: train 3.2554 | val 3.3538
step 333500: train 3.2976 | val 3.3024
step 333600: train 3.2637 | val 3.3250
step 333700: train 3.3241 | val 3.5559
step 333800: train 3.0752 | val 3.3475
step 333900: train 3.1414 | val 3.3535
step 334000: train 3.3381 | val 3.4112
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 334100: train 3.1746 | val 3.3791
step 334200: train 3.2104 | val 3.2840
step 334300: train 3.0692 | val 3.4504
step 334400: train 3.2408 | val 3.2605
step 334500: train 3.2450 | val 3.3020
step 334600: train 2.9912 | val 3.3410
step 334700: train 3.1317 | val 3.3838
Resolving data files: 100%
 2410/2410 [00:00<00:00, 24308.37it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8999.00it/s]
step 334800: train 3.3713 | val 3.2958
step 334900: train 3.1400 | val 3.3545
step 335000: train 3.2275 | val 3.4936
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 335100: train 2.9472 | val 3.3845
step 335200: train 3.5202 | val 3.3921
step 335300: train 3.1447 | val 3.4162
step 335400: train 3.1035 | val 3.4982
step 335500: train 3.2540 | val 3.2595
step 335600: train 3.4831 | val 3.3987
step 335700: train 3.1834 | val 3.3512
step 335800: train 3.1331 | val 3.3419
step 335900: train 3.0776 | val 3.3123
step 336000: train 3.1309 | val 3.3078
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25021.03it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8554.44it/s]
step 336100: train 3.4916 | val 3.3545
step 336200: train 3.4401 | val 3.2936
step 336300: train 3.3468 | val 3.3164
step 336400: train 3.0679 | val 3.5376
step 336500: train 3.2392 | val 3.3292
step 336600: train 3.4736 | val 3.3328
step 336700: train 3.2947 | val 3.3839
step 336800: train 3.3057 | val 3.3403
step 336900: train 3.3110 | val 3.2454
step 337000: train 3.4600 | val 3.4266
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 337100: train 3.3438 | val 3.2362
step 337200: train 3.4123 | val 3.2636
step 337300: train 3.3735 | val 3.3215
step 337400: train 3.2202 | val 3.3494
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35550.20it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11400.88it/s]
step 337500: train 3.3652 | val 3.2567
step 337600: train 3.3213 | val 3.3083
step 337700: train 3.4341 | val 3.4537
step 337800: train 3.5321 | val 3.3431
step 337900: train 3.2402 | val 3.3534
step 338000: train 3.2693 | val 3.3702
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 338100: train 3.2014 | val 3.4434
step 338200: train 3.3408 | val 3.2207
step 338300: train 3.3124 | val 3.3580
step 338400: train 3.2340 | val 3.3219
step 338500: train 3.7626 | val 3.3012
step 338600: train 3.3373 | val 3.2769
step 338700: train 3.3946 | val 3.2743
Resolving data files: 100%
 2410/2410 [00:00<00:00, 31445.08it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 18387.43it/s]
step 338800: train 3.2646 | val 3.3355
step 338900: train 3.2497 | val 3.2776
step 339000: train 3.4112 | val 3.3057
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 339100: train 3.2568 | val 3.5330
step 339200: train 3.1250 | val 3.3237
step 339300: train 3.2276 | val 3.3378
step 339400: train 3.2096 | val 3.3807
step 339500: train 3.3509 | val 3.3481
step 339600: train 3.4010 | val 3.2794
step 339700: train 3.4724 | val 3.4413
step 339800: train 3.2321 | val 3.2432
step 339900: train 3.3314 | val 3.2748
step 340000: train 3.3311 | val 3.3184
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 340100: train 3.2510 | val 3.3663
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28347.84it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8118.49it/s]
step 340200: train 3.4172 | val 3.2721
step 340300: train 3.3594 | val 3.3304
step 340400: train 3.2669 | val 3.4669
step 340500: train 3.3370 | val 3.3594
step 340600: train 3.2597 | val 3.3674
step 340700: train 3.6694 | val 3.3880
step 340800: train 3.4092 | val 3.4668
step 340900: train 3.3042 | val 3.2359
step 341000: train 3.2468 | val 3.3668
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 341100: train 3.2673 | val 3.3213
step 341200: train 3.1566 | val 3.3211
step 341300: train 3.3322 | val 3.2836
step 341400: train 3.2801 | val 3.2852
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40817.59it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8428.46it/s]
step 341500: train 3.0905 | val 3.3579
step 341600: train 3.2996 | val 3.3007
step 341700: train 3.2306 | val 3.3344
step 341800: train 3.9265 | val 3.5656
step 341900: train 3.3518 | val 3.3582
step 342000: train 3.1209 | val 3.3610
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 342100: train 3.0375 | val 3.4128
step 342200: train 2.9375 | val 3.3925
step 342300: train 3.0351 | val 3.2884
step 342400: train 3.0475 | val 3.4560
step 342500: train 3.1271 | val 3.2611
step 342600: train 3.2388 | val 3.3099
step 342700: train 3.3343 | val 3.3445
step 342800: train 3.2292 | val 3.3927
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38712.99it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16903.75it/s]
step 342900: train 3.1000 | val 3.2939
step 343000: train 2.9519 | val 3.3660
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 343100: train 3.3267 | val 3.5018
step 343200: train 3.1207 | val 3.3867
step 343300: train 3.3162 | val 3.3954
step 343400: train 3.3492 | val 3.4042
step 343500: train 3.4671 | val 3.4774
step 343600: train 3.6576 | val 3.2263
step 343700: train 3.5816 | val 3.3679
step 343800: train 3.3058 | val 3.3192
step 343900: train 3.3500 | val 3.3114
step 344000: train 3.2463 | val 3.2708
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 344100: train 3.2118 | val 3.2693
Resolving data files: 100%
 2410/2410 [00:00<00:00, 34884.74it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9493.36it/s]
step 344200: train 3.4194 | val 3.3326
step 344300: train 3.3980 | val 3.2693
step 344400: train 3.3515 | val 3.2950
step 344500: train 3.3525 | val 3.5325
step 344600: train 3.2764 | val 3.3197
step 344700: train 3.2772 | val 3.3249
step 344800: train 3.2313 | val 3.3459
step 344900: train 3.9935 | val 3.3364
step 345000: train 3.4983 | val 3.2522
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 345100: train 3.4039 | val 3.4210
step 345200: train 3.1744 | val 3.2221
step 345300: train 3.2004 | val 3.2607
step 345400: train 3.2036 | val 3.3038
step 345500: train 3.2155 | val 3.3367
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39483.74it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11220.72it/s]
step 345600: train 3.3030 | val 3.2457
step 345700: train 3.1827 | val 3.2974
step 345800: train 3.4844 | val 3.4353
step 345900: train 3.4261 | val 3.3326
step 346000: train 3.3418 | val 3.3419
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 346100: train 3.3982 | val 3.3692
step 346200: train 3.5076 | val 3.4358
step 346300: train 3.2175 | val 3.2131
step 346400: train 3.3549 | val 3.3490
step 346500: train 3.2883 | val 3.2981
step 346600: train 3.2790 | val 3.2967
step 346700: train 3.2424 | val 3.2796
step 346800: train 3.3800 | val 3.2665
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42939.19it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8519.32it/s]
step 346900: train 3.2495 | val 3.3271
step 347000: train 3.3087 | val 3.2736
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 347100: train 3.3455 | val 3.2998
step 347200: train 3.3741 | val 3.5283
step 347300: train 3.3099 | val 3.3163
step 347400: train 3.2941 | val 3.3233
step 347500: train 3.3001 | val 3.3740
step 347600: train 3.4104 | val 3.3371
step 347700: train 3.4694 | val 3.2485
step 347800: train 3.1780 | val 3.4247
step 347900: train 3.4173 | val 3.2357
step 348000: train 3.2785 | val 3.2807
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 348100: train 3.3683 | val 3.3158
step 348200: train 3.0822 | val 3.3526
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27462.09it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 16043.79it/s]
step 348300: train 3.1660 | val 3.2603
step 348400: train 3.2869 | val 3.3240
step 348500: train 3.5804 | val 3.4576
step 348600: train 3.3563 | val 3.3533
step 348700: train 3.2193 | val 3.3590
step 348800: train 3.3228 | val 3.3810
step 348900: train 3.1715 | val 3.4575
step 349000: train 3.2064 | val 3.2290
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 349100: train 3.0820 | val 3.3747
step 349200: train 3.2164 | val 3.3194
step 349300: train 3.3708 | val 3.3131
step 349400: train 3.2883 | val 3.2849
step 349500: train 2.9308 | val 3.2875
Resolving data files: 100%
 2410/2410 [00:00<00:00, 43878.81it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8770.76it/s]
step 349600: train 3.1453 | val 3.3474
step 349700: train 3.1660 | val 3.3008
step 349800: train 3.1688 | val 3.3334
step 349900: train 3.0826 | val 3.5555
step 350000: train 3.1430 | val 3.3411
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 350100: train 3.5776 | val 3.3488
step 350200: train 3.1924 | val 3.4014
step 350300: train 3.3494 | val 3.3733
step 350400: train 3.3788 | val 3.2715
step 350500: train 2.7931 | val 3.4487
step 350600: train 3.2880 | val 3.2538
step 350700: train 3.3935 | val 3.2988
step 350800: train 2.8511 | val 3.3319
step 350900: train 3.1850 | val 3.3880
Resolving data files: 100%
 2410/2410 [00:00<00:00, 45201.28it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17172.18it/s]
step 351000: train 3.1020 | val 3.2779
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 351100: train 3.2364 | val 3.3448
step 351200: train 3.2506 | val 3.4846
step 351300: train 3.1034 | val 3.3709
step 351400: train 3.2949 | val 3.3691
step 351500: train 3.4721 | val 3.3844
step 351600: train 3.2528 | val 3.4454
step 351700: train 3.3129 | val 3.2134
step 351800: train 3.3135 | val 3.3544
step 351900: train 3.4324 | val 3.3024
step 352000: train 3.3109 | val 3.2919
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 352100: train 3.3147 | val 3.2707
step 352200: train 3.3402 | val 3.2677
Resolving data files: 100%
 2410/2410 [00:00<00:00, 33911.39it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8430.28it/s]
step 352300: train 3.2148 | val 3.3160
step 352400: train 3.2937 | val 3.2574
step 352500: train 3.3916 | val 3.2874
step 352600: train 3.2259 | val 3.5036
step 352700: train 3.7408 | val 3.3085
step 352800: train 3.4018 | val 3.3122
step 352900: train 3.3125 | val 3.3553
step 353000: train 3.3203 | val 3.3260
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 353100: train 3.2351 | val 3.2243
step 353200: train 3.2676 | val 3.4128
step 353300: train 3.2888 | val 3.2151
step 353400: train 3.3723 | val 3.2594
step 353500: train 3.3098 | val 3.2911
step 353600: train 3.3851 | val 3.3369
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35341.26it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13213.97it/s]
step 353700: train 3.3131 | val 3.2391
step 353800: train 3.2738 | val 3.2993
step 353900: train 3.2040 | val 3.4357
step 354000: train 3.3861 | val 3.3371
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 354100: train 3.3664 | val 3.3449
step 354200: train 3.1615 | val 3.3632
step 354300: train 3.2675 | val 3.4328
step 354400: train 3.2834 | val 3.2067
step 354500: train 3.4553 | val 3.3443
step 354600: train 3.4885 | val 3.3031
step 354700: train 3.3962 | val 3.2903
step 354800: train 3.2728 | val 3.2656
step 354900: train 3.4162 | val 3.2717
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40434.55it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 12443.10it/s]
step 355000: train 3.3010 | val 3.3151
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 355100: train 3.1593 | val 3.2708
step 355200: train 3.4681 | val 3.2941
step 355300: train 3.3199 | val 3.5152
step 355400: train 3.2301 | val 3.3154
step 355500: train 3.2119 | val 3.3269
step 355600: train 3.5596 | val 3.3926
step 355700: train 3.3662 | val 3.3483
step 355800: train 3.3571 | val 3.2555
step 355900: train 3.1124 | val 3.4231
step 356000: train 3.1978 | val 3.2297
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 356100: train 3.4041 | val 3.2728
step 356200: train 3.2342 | val 3.3038
step 356300: train 3.4007 | val 3.3529
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28441.72it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 11474.18it/s]
step 356400: train 3.2863 | val 3.2641
step 356500: train 3.2125 | val 3.3251
step 356600: train 3.1641 | val 3.4609
step 356700: train 3.1932 | val 3.3544
step 356800: train 3.5491 | val 3.3615
step 356900: train 3.3738 | val 3.3852
step 357000: train 3.3491 | val 3.4633
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 357100: train 3.1884 | val 3.2336
step 357200: train 3.1910 | val 3.3704
step 357300: train 3.1601 | val 3.3241
step 357400: train 3.1331 | val 3.3147
step 357500: train 3.2626 | val 3.2828
step 357600: train 3.4052 | val 3.2873
Resolving data files: 100%
 2410/2410 [00:00<00:00, 41239.57it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8676.16it/s]
step 357700: train 3.0057 | val 3.3498
step 357800: train 3.0821 | val 3.2943
step 357900: train 3.0614 | val 3.3217
step 358000: train 3.1719 | val 3.5476
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 358100: train 3.2297 | val 3.3412
step 358200: train 3.0834 | val 3.3456
step 358300: train 3.1888 | val 3.4004
step 358400: train 3.1493 | val 3.3731
step 358500: train 3.3881 | val 3.2739
step 358600: train 3.3911 | val 3.4395
step 358700: train 3.4901 | val 3.2515
step 358800: train 3.3631 | val 3.2925
step 358900: train 3.3867 | val 3.3112
step 359000: train 3.5231 | val 3.3499
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25254.77it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8696.72it/s]
step 359100: train 3.6484 | val 3.2575
step 359200: train 3.3539 | val 3.3089
step 359300: train 3.3788 | val 3.4414
step 359400: train 3.7625 | val 3.3381
step 359500: train 3.2839 | val 3.3456
step 359600: train 3.3513 | val 3.3706
step 359700: train 3.2254 | val 3.4255
step 359800: train 3.2320 | val 3.1991
step 359900: train 3.4840 | val 3.3397
step 360000: train 3.2933 | val 3.3103
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 360100: train 3.3903 | val 3.2781
step 360200: train 3.3519 | val 3.2506
step 360300: train 3.1010 | val 3.2515
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42980.64it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9905.07it/s]
step 360400: train 3.3361 | val 3.3168
step 360500: train 3.2030 | val 3.2580
step 360600: train 3.2490 | val 3.2817
step 360700: train 3.1143 | val 3.5063
step 360800: train 3.3945 | val 3.3129
step 360900: train 3.3184 | val 3.3176
step 361000: train 3.3702 | val 3.3632
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 361100: train 3.1524 | val 3.3395
step 361200: train 3.4698 | val 3.2304
step 361300: train 3.4663 | val 3.4118
step 361400: train 3.3636 | val 3.2131
step 361500: train 3.4131 | val 3.2540
step 361600: train 3.2204 | val 3.2948
step 361700: train 3.3519 | val 3.3343
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26375.76it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8339.05it/s]
step 361800: train 3.2427 | val 3.2432
step 361900: train 3.2945 | val 3.3055
step 362000: train 3.2290 | val 3.4335
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 362100: train 3.3726 | val 3.3377
step 362200: train 3.2287 | val 3.3373
step 362300: train 3.1093 | val 3.3631
step 362400: train 3.4054 | val 3.4350
step 362500: train 3.2739 | val 3.2080
step 362600: train 3.3705 | val 3.3639
step 362700: train 3.4397 | val 3.3069
step 362800: train 3.1850 | val 3.2963
step 362900: train 3.2682 | val 3.2653
step 363000: train 3.2703 | val 3.2654
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39784.60it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15622.08it/s]
step 363100: train 3.2729 | val 3.3291
step 363200: train 3.1606 | val 3.2770
step 363300: train 3.3449 | val 3.2989
step 363400: train 3.2169 | val 3.5289
step 363500: train 3.2372 | val 3.3235
step 363600: train 3.2525 | val 3.3205
step 363700: train 3.1702 | val 3.3753
step 363800: train 3.3853 | val 3.3472
step 363900: train 3.2164 | val 3.2519
step 364000: train 3.3053 | val 3.4137
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 364100: train 3.2805 | val 3.2253
step 364200: train 3.2300 | val 3.2668
step 364300: train 3.2867 | val 3.3046
step 364400: train 3.1795 | val 3.3595
Resolving data files: 100%
 2410/2410 [00:00<00:00, 46501.33it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8430.88it/s]
step 364500: train 3.2547 | val 3.2747
step 364600: train 2.9647 | val 3.3352
step 364700: train 3.0765 | val 3.4675
step 364800: train 2.9739 | val 3.3614
step 364900: train 3.0823 | val 3.3748
step 365000: train 3.3505 | val 3.3928
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 365100: train 2.8773 | val 3.4797
step 365200: train 3.2231 | val 3.2462
step 365300: train 2.9512 | val 3.3780
step 365400: train 2.9686 | val 3.3275
step 365500: train 2.7710 | val 3.3256
step 365600: train 3.3007 | val 3.2921
step 365700: train 2.8330 | val 3.2886
Resolving data files: 100%
 2410/2410 [00:00<00:00, 28260.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 6474.12it/s]
step 365800: train 3.0879 | val 3.3540
step 365900: train 3.1176 | val 3.3043
step 366000: train 3.0775 | val 3.3332
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 366100: train 2.9606 | val 3.5542
step 366200: train 3.3860 | val 3.3366
step 366300: train 3.3976 | val 3.3269
step 366400: train 3.4517 | val 3.3765
step 366500: train 3.2993 | val 3.3320
step 366600: train 3.3269 | val 3.2635
step 366700: train 3.2483 | val 3.4123
step 366800: train 3.3675 | val 3.2160
step 366900: train 3.1677 | val 3.2500
step 367000: train 3.2444 | val 3.2920
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 367100: train 3.3294 | val 3.3317
Resolving data files: 100%
 2410/2410 [00:00<00:00, 26632.82it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17216.48it/s]
step 367200: train 3.1873 | val 3.2444
step 367300: train 3.2552 | val 3.2983
step 367400: train 3.2770 | val 3.4275
step 367500: train 3.3298 | val 3.3267
step 367600: train 3.3450 | val 3.3259
step 367700: train 3.1829 | val 3.3423
step 367800: train 3.2065 | val 3.4159
step 367900: train 3.2690 | val 3.1887
step 368000: train 3.3569 | val 3.3352
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 368100: train 3.2399 | val 3.2944
step 368200: train 3.4904 | val 3.2742
step 368300: train 3.3711 | val 3.2350
step 368400: train 3.2475 | val 3.2388
Resolving data files: 100%
 2410/2410 [00:00<00:00, 38218.70it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 7954.52it/s]
step 368500: train 3.2798 | val 3.3078
step 368600: train 3.2904 | val 3.2546
step 368700: train 3.2404 | val 3.2808
step 368800: train 3.2746 | val 3.4879
step 368900: train 3.2557 | val 3.3004
step 369000: train 3.2512 | val 3.3051
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 369100: train 3.2595 | val 3.3453
step 369200: train 3.3185 | val 3.3162
step 369300: train 3.3439 | val 3.2294
step 369400: train 3.3468 | val 3.4084
step 369500: train 3.4462 | val 3.2110
step 369600: train 3.2662 | val 3.2540
step 369700: train 3.4020 | val 3.2767
step 369800: train 3.4315 | val 3.3238
Resolving data files: 100%
 2410/2410 [00:00<00:00, 42609.41it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15591.38it/s]
step 369900: train 3.2241 | val 3.2479
step 370000: train 3.3418 | val 3.2960
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 370100: train 3.5285 | val 3.4276
step 370200: train 3.2285 | val 3.3223
step 370300: train 3.2029 | val 3.3335
step 370400: train 3.2753 | val 3.3538
step 370500: train 3.3836 | val 3.4264
step 370600: train 3.5290 | val 3.2025
step 370700: train 3.3128 | val 3.3437
step 370800: train 3.4861 | val 3.2993
step 370900: train 3.2654 | val 3.2837
step 371000: train 3.1378 | val 3.2541
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 371100: train 3.2257 | val 3.2574
Resolving data files: 100%
 2410/2410 [00:00<00:00, 40803.09it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 9594.97it/s]
step 371200: train 3.3109 | val 3.3214
step 371300: train 3.3132 | val 3.2714
step 371400: train 3.2279 | val 3.2950
step 371500: train 3.4984 | val 3.5316
step 371600: train 3.3017 | val 3.3156
step 371700: train 3.1258 | val 3.3192
step 371800: train 3.3648 | val 3.3680
step 371900: train 3.2369 | val 3.3392
step 372000: train 3.1791 | val 3.2498
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 372100: train 3.3719 | val 3.4234
step 372200: train 3.1342 | val 3.2262
step 372300: train 3.2426 | val 3.2602
step 372400: train 3.3908 | val 3.2875
step 372500: train 3.2383 | val 3.3513
Resolving data files: 100%
 2410/2410 [00:00<00:00, 35803.93it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8289.26it/s]
step 372600: train 3.1928 | val 3.2564
step 372700: train 3.4952 | val 3.3245
step 372800: train 3.0620 | val 3.4509
step 372900: train 3.2445 | val 3.3551
step 373000: train 3.1565 | val 3.3665
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 373100: train 3.4313 | val 3.3862
step 373200: train 3.1522 | val 3.4560
step 373300: train 3.2107 | val 3.2231
step 373400: train 3.2047 | val 3.3679
step 373500: train 3.0243 | val 3.3182
step 373600: train 3.4209 | val 3.3128
step 373700: train 3.1924 | val 3.2722
step 373800: train 3.0340 | val 3.2749
Resolving data files: 100%
 2410/2410 [00:00<00:00, 39800.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 13863.18it/s]
step 373900: train 3.1718 | val 3.3384
step 374000: train 2.9576 | val 3.2896
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 374100: train 3.0244 | val 3.3118
step 374200: train 3.0414 | val 3.5398
step 374300: train 3.2512 | val 3.3376
step 374400: train 3.3923 | val 3.3134
step 374500: train 3.3713 | val 3.3617
step 374600: train 3.3612 | val 3.3289
step 374700: train 3.6137 | val 3.2404
step 374800: train 3.5025 | val 3.4035
step 374900: train 3.1237 | val 3.2102
step 375000: train 3.4813 | val 3.2425
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 375100: train 3.6173 | val 3.2806
step 375200: train 3.2466 | val 3.3123
Resolving data files: 100%
 2410/2410 [00:00<00:00, 32682.92it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 15590.14it/s]
step 375300: train 3.3324 | val 3.2287
step 375400: train 3.2576 | val 3.2761
step 375500: train 3.3134 | val 3.4140
step 375600: train 3.2906 | val 3.3141
step 375700: train 3.1130 | val 3.3217
step 375800: train 3.3787 | val 3.3298
step 375900: train 3.4350 | val 3.4064
step 376000: train 3.1609 | val 3.1878
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 376100: train 3.2565 | val 3.3312
step 376200: train 3.3302 | val 3.2906
step 376300: train 3.3168 | val 3.2629
step 376400: train 3.2126 | val 3.2319
step 376500: train 3.3027 | val 3.2370
Resolving data files: 100%
 2410/2410 [00:00<00:00, 25886.06it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 8576.18it/s]
step 376600: train 3.4167 | val 3.2994
step 376700: train 3.3485 | val 3.2473
step 376800: train 3.6116 | val 3.2663
step 376900: train 3.3251 | val 3.4957
step 377000: train 3.2492 | val 3.2931
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 377100: train 3.2031 | val 3.2990
step 377200: train 3.2481 | val 3.3425
step 377300: train 3.2811 | val 3.3185
step 377400: train 3.2648 | val 3.2281
step 377500: train 3.5257 | val 3.3995
step 377600: train 3.2518 | val 3.1995
step 377700: train 3.2664 | val 3.2393
step 377800: train 3.1314 | val 3.2684
step 377900: train 3.2741 | val 3.3164
Resolving data files: 100%
 2410/2410 [00:00<00:00, 36291.11it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 14806.29it/s]
step 378000: train 3.2176 | val 3.2305
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 378100: train 3.1669 | val 3.2889
step 378200: train 3.1611 | val 3.4226
step 378300: train 3.3220 | val 3.3264
step 378400: train 3.3437 | val 3.3349
step 378500: train 3.2898 | val 3.3525
step 378600: train 3.0079 | val 3.4331
step 378700: train 3.1950 | val 3.2024
step 378800: train 3.2041 | val 3.3439
step 378900: train 3.2727 | val 3.2940
step 379000: train 3.2893 | val 3.2909
Saved checkpoint: /workspace/chkpt/tinygpt_latest.pt
step 379100: train 3.1918 | val 3.2511
step 379200: train 3.2247 | val 3.2526
Resolving data files: 100%
 2410/2410 [00:00<00:00, 27348.56it/s]
Resolving data files: 100%
 140/140 [00:00<00:00, 17009.52it/s]
step 379300: train 3.2773 | val 3.3228
step 379400: train 3.1653 | val 3.2678
