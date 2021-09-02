<p align="right">Updated Date: AUG 31, 2021</p>


预处理及数据准备
1. 对三分量波形重采样至100Hz
1. 去线性趋势
2. 带通滤波：1～15Hz
3. 拾取P波初至
4. 截P波初至前2s，后28s波形，共3000个采样点


输入
* 三分量波形(按E，N，Z顺序): 3000*3
* 输入形状: (3000, 1, 3)，3000表示height，1表示width, 3表示channel


输出
* 概率向量：(eq, qb)，eq表示模型预测该波形是天然地震的概率，qb表示预测是爆破事件的概率，二者之和是1。原则上eq>0.5可以认为是地震，出于实际需要，也可以设置不同的阈值。
