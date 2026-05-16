## Raytracing in One Weekend に One Weekend 以上かけてノロノロと取り組む

**記録**

### 1日目  
画像を表示しただけ  
![画像表示](week1/day1.png)

### 2日目  
vec3, ray クラスなどを定義しただけ  

### 3日目  
ビューポートのy座標に基づいてグラデーションを画像を表示  
![グラデーション画像](week1/day3_gradation.png)

### 4日目  
レイと球の交差判定をして、ビューポートに投影  
![球のシルエット投影](week1/day4_silhouette.png)  
<br>
法線を計算して可視化  
![球のシルエット投影](week1/day4_normal.png)  

### 5日目
衝突可能オブジェクトの抽象化, 球の位置の可視化<br>
![級の位置と法線の可視化](week1/day5.png)
<br>

カメラクラスを定義

アンチエイリアス<br>
<img src="week1/day6_anti_alias.png" alt="アンチエイリアス" width="480">

拡散反射球の表示<br>
<img src="week1/day6_diffuse.png" alt="拡散反射球" width="480">


### 6日目

ガンマ補正適用<br>
<img src="week1/day6_diffuse_gamma.png" alt="ガンマ補正後" width="480">

シャドウアクネ削除<br>
<img src="week1/day6_diffuse_gamma_acne.png" alt="シャドウアクネ削除" width="480">