# MagicFormula
A python implementation of the Magic Formula Investing described in The Little Book That Still Beats the Market by investor Joel Greenblatt.

Stock data is retrieved using the yahooquery. https://yahooquery.dpguthrie.com

## Getting started

Here is how to download the source code, install all dependencies, and run the program:

```
git clone https://github.com/hsinming/MagicFormula.git
cd MagicFormula
pip install -r requirements.txt
python magic_formula.py
```

## Configuration

There are three flags to be aware of:

* `-c`    Country. ['US', 'TW'], default='US'
* `-t`    Thread number, default=1
* `-b`    Batch size, default=1
* `-m`    Minimal market cap
* `--price`    forcefully renew price
* `--key-stats`    forcefully renew key stats
* `--profile`    forcefully renew summary profile
* `--financial`    forcefully renew financial statements

## Inspired by nblade66 [site](https://github.com/nblade66/MagicFormula)
