{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "2ca00e30",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "855818eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "test=pd.read_csv(r'C:\\Users\\tusha\\Documents\\test.csv')\n",
    "train=pd.read_csv(r'C:\\Users\\tusha\\Documents\\train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "193b23e6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "y_train=train[\"is_anomaly\"]\n",
    "y_lab=LabelEncoder()\n",
    "y_train=y_lab.fit_transform(y_train)\n",
    "y_train\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9c24a238",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>timestamp</th>\n",
       "      <th>value</th>\n",
       "      <th>predicted</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1396332000</td>\n",
       "      <td>20.00000</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1396332300</td>\n",
       "      <td>20.00000</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1396332600</td>\n",
       "      <td>20.00000</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1396332900</td>\n",
       "      <td>20.00000</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1396333200</td>\n",
       "      <td>20.00000</td>\n",
       "      <td>20.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3955</th>\n",
       "      <td>1397518500</td>\n",
       "      <td>20.00384</td>\n",
       "      <td>19.836240</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3956</th>\n",
       "      <td>1397518800</td>\n",
       "      <td>20.00384</td>\n",
       "      <td>19.207998</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3957</th>\n",
       "      <td>1397519100</td>\n",
       "      <td>20.00384</td>\n",
       "      <td>20.103437</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3958</th>\n",
       "      <td>1397519400</td>\n",
       "      <td>20.00384</td>\n",
       "      <td>19.346764</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3959</th>\n",
       "      <td>1397519700</td>\n",
       "      <td>20.00384</td>\n",
       "      <td>20.134947</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3960 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       timestamp     value  predicted\n",
       "0     1396332000  20.00000  20.000000\n",
       "1     1396332300  20.00000  20.000000\n",
       "2     1396332600  20.00000  20.000000\n",
       "3     1396332900  20.00000  20.000000\n",
       "4     1396333200  20.00000  20.000000\n",
       "...          ...       ...        ...\n",
       "3955  1397518500  20.00384  19.836240\n",
       "3956  1397518800  20.00384  19.207998\n",
       "3957  1397519100  20.00384  20.103437\n",
       "3958  1397519400  20.00384  19.346764\n",
       "3959  1397519700  20.00384  20.134947\n",
       "\n",
       "[3960 rows x 3 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "752adaf2",
   "metadata": {},
   "outputs": [],
   "source": [
    "features=[\"value\",\"predicted\"]\n",
    "x_train=train[features]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b88237ee",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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qir/61a5Fl9//bXuuOMODR06VL169dLpp5+uX//61/J4PP4yXfVa/P3vf9ell16qvLw82Ww2vfLKKwGvJ/J779mzR5deeqn69OmjAQMGaP78+WpqaorH1w4q3LU4ceKEfvazn2nUqFHq06eP8vLydM0112j//v0B5+gK16Ktn4nm/vu//1s2m01LliwJOJ5y18EgYsuXLzfdu3c3Tz75pHn//fdNaWmp6dOnj/n000+TXbV2u+iii8xTTz1ldu7caaqrq80ll1xiBg0aZI4ePeovc88995jMzEzz8ssvm3fffddcccUVJjc317hcLn+ZH//4x2bgwIFmzZo1ZseOHWbSpEnm29/+tvn666/9ZYqKikxhYaHZtGmT2bRpkyksLDTTpk1L6PeN1LZt28yQIUPMv/3bv5nS0lL/catciy+++MIMHjzYXHvttWbr1q2mpqbGrF271nzyySf+Mla5Fnfffbfp37+/+ctf/mJqamrMSy+9ZE455RSzZMkSf5muei1WrVplbr/9dvPyyy8bSWblypUBryfqe3/99demsLDQTJo0yezYscOsWbPG5OXlmblz58b9GviEuxaHDx82U6ZMMS+88IL54IMPzObNm82YMWPM6NGjA87RFa5FWz8TPitXrjTf/va3TV5ennnwwQcDXku160AQisK///u/mx//+McBx84880zz85//PEk1ir0DBw4YSWbDhg3GGGM8Ho/Jyckx99xzj7/M8ePHjcPhMI8//rgxxvuPQPfu3c3y5cv9ZWpra01aWpqpqKgwxhjz/vvvG0lmy5Yt/jKbN282kswHH3yQiK8WsSNHjphhw4aZNWvWmAsuuMAfhKx0LX72s5+Z888/P+TrVroWl1xyifnP//zPgGPFxcXm6quvNsZY51q0vOkl8nuvWrXKpKWlmdraWn+Z559/3mRkZBin0xmX7xtOuADgs23bNiPJ/4tyV7wWoa7Dvn37zMCBA83OnTvN4MGDA4JQKl4HusYi1NTUpKqqKk2dOjXg+NSpU7Vp06Yk1Sr2nE6nJKlfv36SpJqaGtXX1wd874yMDF1wwQX+711VVaUTJ04ElMnLy1NhYaG/zObNm+VwODRmzBh/mbFjx8rhcKTc9bvxxht1ySWXaMqUKQHHrXQtXn31VZ177rn6wQ9+oKysLJ199tl68skn/a9b6Vqcf/75+tvf/qaP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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.plot(x_train[\"value\"][y_train==0],x_train[\"predicted\"][y_train==0],\"b^\")\n",
    "plt.plot(x_train[\"value\"][y_train==1],x_train[\"predicted\"][y_train==1],\"go\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "65035344",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestClassifier(n_jobs=-1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(n_jobs=-1)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestClassifier(n_jobs=-1)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rnd_clf=RandomForestClassifier(n_jobs=-1)\n",
    "rnd_clf.fit(x_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "6cb23d51",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_test=test[features]\n",
    "prediction=rnd_clf.predict(x_test)\n",
    "prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "90f8b1fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>timestamp</th>\n",
       "      <th>is_anomaly</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1425008573</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1425008873</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1425009173</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1425009473</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1425009773</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15825</th>\n",
       "      <td>1429756073</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15826</th>\n",
       "      <td>1429756373</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15827</th>\n",
       "      <td>1429756673</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15828</th>\n",
       "      <td>1429756973</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15829</th>\n",
       "      <td>1429757273</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>15830 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        timestamp  is_anomaly\n",
       "0      1425008573       False\n",
       "1      1425008873       False\n",
       "2      1425009173       False\n",
       "3      1425009473       False\n",
       "4      1425009773       False\n",
       "...           ...         ...\n",
       "15825  1429756073       False\n",
       "15826  1429756373       False\n",
       "15827  1429756673       False\n",
       "15828  1429756973       False\n",
       "15829  1429757273       False\n",
       "\n",
       "[15830 rows x 2 columns]"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "submission=pd.read_csv(r'C:\\Users\\tusha\\Documents\\Submission.csv')\n",
    "submission"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "ebbb4338",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False, False, False, ..., False, False, False])"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "inv_transform=y_lab.inverse_transform(prediction)\n",
    "inv_transform"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "d6f9bbea",
   "metadata": {},
   "outputs": [],
   "source": [
    "data={\"timestamp\":[],\"is_anomaly\":[]}\n",
    "for id,pred in zip(test[\"timestamp\"].unique(),inv_transform):\n",
    "  data[\"timestamp\"].append(id)\n",
    "  data[\"is_anomaly\"].append(pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "2a40dc9f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>timestamp</th>\n",
       "      <th>is_anomaly</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1396332000</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1396332300</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1396332600</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1396332900</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1396333200</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3955</th>\n",
       "      <td>1397518500</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3956</th>\n",
       "      <td>1397518800</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3957</th>\n",
       "      <td>1397519100</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3958</th>\n",
       "      <td>1397519400</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3959</th>\n",
       "      <td>1397519700</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>3960 rows × 2 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       timestamp  is_anomaly\n",
       "0     1396332000       False\n",
       "1     1396332300       False\n",
       "2     1396332600       False\n",
       "3     1396332900       False\n",
       "4     1396333200       False\n",
       "...          ...         ...\n",
       "3955  1397518500       False\n",
       "3956  1397518800       False\n",
       "3957  1397519100       False\n",
       "3958  1397519400       False\n",
       "3959  1397519700       False\n",
       "\n",
       "[3960 rows x 2 columns]"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results=pd.DataFrame(data,columns=[\"timestamp\",\"is_anomaly\"])\n",
    "results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fc388ec3",
   "metadata": {},
   "outputs": [],
   "source": [
    "results.to_csv(\"submission.csv\",index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d4e8477",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
