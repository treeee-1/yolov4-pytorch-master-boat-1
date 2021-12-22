import json

q = input()
w = input()
a = {
        "boat": {
            "path": "F:/Data/ShuiKu_GH/2020-1/Boat",
            "coods": [
                {
                    "id": "001",
                    "lon": q,
                    "lat": w
                },
                {
                    "id": "002",
                    "lon": 7314.6427126249,
                    "lat": 4399.70037276876
                }
            ]
        }
    },

f2 = open('new_json.json', 'w',encoding='utf-8')
b = json.dumps(a, indent = 2, sort_keys = False, ensure_ascii = False)

f2.write(b)
f2.close()