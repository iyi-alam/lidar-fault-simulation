# Lookup dict of experimental values observed in the experiment

data = {
    "1u": {
        "T": 99.5,
        "R": 64.9,
        "del_T": 0.5,
        "del_R": 1.5,
        "mode": "light"
    },
    "1l": {
        "T": 99.8,
        "R": 61.9,
        "del_T": 0.3,
        "del_R": 1.5,
        "mode": "light"
    },
    "2u": {
        "T": 95.8,
        "R": 54.9,
        "del_T": 4.3,
        "del_R": 8.5,
        "mode": "moderate"
    },
    "2l": {
        "T": 94.5,
        "R": 52.8,
        "del_T": 5.6,
        "del_R": 10.6,
        "mode": "moderate"
    },
    "3u": {
        "T": 65.8,
        "R": 29.1,
        "del_T": 34.2,
        "del_R": 34.3,
        "mode": "heavy"
    },
    "3l": {
        "T": 65.4,
        "R": 25.9,
        "del_T": 34.6,
        "del_R": 37.5,
        "mode": "heavy"
    },
    "4u": {
        "T": 96.1,
        "R": 57.5,
        "del_T": 3.9,
        "del_R": 5.9,
        "mode": "light"
    },
    "4l": {
        "T": 94.1,
        "R": 62.2,
        "del_T": 6.0,
        "del_R": 1.2,
        "mode": "light"
    },
    "5u": {
        "T": 72.0,
        "R": 55.5,
        "del_T": 28.0,
        "del_R": 7.9,
        "mode": "moderate"
    },
    "5l": {
        "T": 73.9,
        "R": 56.3,
        "del_T": 26.1,
        "del_R": 7.1,
        "mode": "moderate"
    },
    "6u": {
        "T": 95.4,
        "R": 56.0,
        "del_T": 4.6,
        "del_R": 7.4,
        "mode": "moderate"
    },
    "6l": {
        "T": 95.6,
        "R": 52.9,
        "del_T": 4.4,
        "del_R": 10.5,
        "mode": "moderate"
    }
}

samples = list(data.keys())

if __name__ == "__main__":
    print(samples)
