# Image processing

Implementation is divided into 3 steps. After the photo is loaded, it undergoes the following operations sequentially:
1) Constrast Limited Adaptive Histogram Equalization [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization).
2) Median filtering.
3) Segmentation using mean-C thresholding <br>

The details of step 3) can be found in the 2.2 paragraph of the scientific paper mentioned in the references section.

The quality of obtained image is measured by following metrics: accuracy, precision, recall, F1-score.

### References

This solution is based on the following scientific paper:

- Dash, J., Bhoi, A. (2017). "A thresholding based technique to extract retinal blood vessels from fundus images." Veer Surendra Sai University of Technology, Burla, Odisha, India. <a href="https://pdf.sciencedirectassets.com/314569/1-s2.0-S2314728817X00044/1-s2.0-S2314728817300272/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQDrKhVwJ3WoZsLcYnzDdM6fbFyvvNIGiZmaLKGuYS2MPQIgVxz83cGDxEMx1FX41jskwVSUhrF7Q%2BR4ISvZEosU%2FAQqvAUI1%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAFGgwwNTkwMDM1NDY4NjUiDMRIuLHh30wdkov1jCqQBUlxGUkJ75LkN2py%2BXJjKpQeHTMp3G6dmaut5BezEzIcqJp0RRNNk6k%2FBUsOBSsg9ukWCYah%2BEPQDTPhL5stnBFPFvJcGnTuhDfDjjhYFdmN0ho2g7wTa2bx8ayR4SJZjW9uFq3thIIkqpQCdMaWHaGS461HAvw%2BJuAGxd1cmB6LDJcghq2EZSWAk%2FmoCUnSCToE1qZ8hC%2FGP3pcuhXu%2FsWNX92rkIHf2x2xYWnQTYkCwafx2N8zUFeJATycRObxU6VBfDiswqKfaeQtiDId5DP6tqWd8mySr9r%2FM1puoQ8x435yNg8IVm%2BgkVollgc1%2Bo2mAKe54CHsSVJgjN9WMQFK4tOgLHybQHqHPIdJsjuwRUKyw9JkSunvAOzmH2CSVkyXNKsrdcxkQvPh9eHcQBkHM%2FarNRFXbokG74bfhyWkzPReuvtVjcb29DsTzCEsuOT9t0g99ww5CKLkA8wFuE%2BLiOwpST5SBPfaSHgrWB4Rk8tjZGWDnTTL%2FVvysJ32AMSJuVowa1mTxq37zWC66Ef80ZCLbtPiAn84nC3tKRb4Ec2n1V8PkDKilThBo1ivwccTKeuVT0BGisLoRGG7tolXE%2FPEQD%2F8wQQbN8oP%2FDxYQhNmd2SkeIm9iVuF%2Bfy9pUSTcV6iLlvyUoLRCtceSWDDz4pXEobARj2EBgrcoDPMsmtB9S2DLvo4rqW0RIcRk60j959oUwJXtID%2BheZAZo9cBi3TrwHG7IZC6%2BhKlx0P7sNKBsRe8FpkTBZET2WCB41jY6H6RwgA%2BDhMffDGMmO%2Bs%2F4y6FmEWyuVo73vKla1WxEx5AHEkWXSLbm7mMQI2jwggN1QUi0G%2FCONBII%2FLqpi9c3rl%2FGO4Qd6eTHMHNJuMNmDs6MGOrEBXCGOgP7Ns39HLQneWIk%2BzCsJodWz2HKyUb%2BxhLSlRKvpDLvC5mYkziF2e4abwAe14ZWiLEU28T5ofbS4qt68LUCdYJEOUutmAl0j%2BBBxpc0EQuLMs%2BT4wh7%2FcqGT8koxzDnkgC1XR%2F3FbfirKA8Vh%2B8I8QJUN8i2j6MIpUNAotzaPRDTmNCYOzs7UI%2FkQjZVncxSKx%2Foj6R5oQjzJ3NmfbTGbJV74j%2B8PVL6jxMNc76T&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20230523T144700Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY74SQ5AF3%2F20230523%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=94036040d9b7b131129c02dfe435b43674459da3ceb4600fd84c016df8d48a22&hash=66135c72345ded512b9b32e580c21ee18f1ece08240b7396dc43da2218c818a1&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S2314728817300272&tid=spdf-53a7cd9b-7807-485d-b9cb-2c15f308549d&sid=1ed8861f93611544cd3b2bc179ef7f3e5c27gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=160f590404015e5558&rr=7cbe17722e4334a4&cc=pl" >Link</a>.

### Results
<!--<img src="result.png"><br>-->
Accuracy of provided sample result is approximetaly 95.6%.
