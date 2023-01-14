# MagicFormula
A python implementation of the Magic Formula Investing described in [The Little Book That Still Beats the Market](https://www.amazon.com/Little-Still-Market-Books-Profits-ebook/dp/B003VWCQB0) by Joel Greenblatt.

This [site](https://www.valuesignals.com/Glossary/Details/Greenblatt_Magic_Formula/13381) explains how it works.

I also found some glossary at this [site](https://www.quant-investing.com/glossary).

## Getting started

#### Step 1: Install Anaconda [(tutorial)](https://docs.anaconda.com/anaconda/install/index.html)
#### Step 2: 
```
git clone https://github.com/hsinming/MagicFormula.git
cd MagicFormula
conda env create -f environment.yml
conda activate stock
python main.py
```

## Configuration

There are some flags to be aware of:

* `-c`    Country. ['US', 'TW'], default='US'
* `-m`    Minimal market cap in USD, default=1000,000,000 USD
* `--quotes`    forcefully renew quotes
* `--profile`    forcefully renew summary profile
* `--financial`    forcefully renew financial statements

### Inspired by nblade66 [(github)](https://github.com/nblade66/MagicFormula)
