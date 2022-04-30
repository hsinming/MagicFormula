# MagicFormula
A python implementation of the Magic Formula Investing described in [The Little Book That Still Beats the Market](https://www.amazon.com/Little-Still-Market-Books-Profits-ebook/dp/B003VWCQB0) by investor Joel Greenblatt.

Stock data is retrieved using the yahooquery. https://yahooquery.dpguthrie.com

## Getting started

### Install Anaconda [tutorial](https://docs.anaconda.com/anaconda/install/index.html)

```
git clone https://github.com/hsinming/MagicFormula.git
cd MagicFormula
conda env create -f environment.yml
conda activate stock
python main.py
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

### Inspired by nblade66 [github repository](https://github.com/nblade66/MagicFormula)
