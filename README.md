# Alpaca Autonomous Trading Agent

Plne autonomni, prubezne samoucici se trading bot pro americke akcie, napojeny
**vyhradne na burzu Alpaca** (paper i live ucty). Bot sam stahuje trzni data,
pocita technicke indikatory, rozhoduje se pomoci online (inkrementalniho)
ML modelu kombinovaneho s pravidlovymi strategiemi, riadi risk management a
sam odesila/spravuje ordery - bez rucniho zasahu za behu.

> **Disclaimer:** Toto je software pro automatizovane obchodovani na financnich
> trzich. Obchodovani s sebou nese riziko ztraty investovaneho kapitalu.
> Vzdy nejdriv dukladne testujte na **paper uctu** (`ALPACA_PAPER=true`) a
> spustte `scripts/run_backtest.py` na historickych datech, nez cokoliv
> pustite naostro. Autor nenese odpovednost za financni ztraty zpusobene
> pouzitim tohoto software.

## Jak funguje "samouceni"

Bot se uci na **dvou urovnich soucasne**, obe prubezne, bez nutnosti rucniho
retrainu:

1. **Online ML model** (`trading_agent/model/online_model.py`) - misto
   jednorazoveho natrenovani modelu, ktery se pak jen pouziva, pobezi
   `river.tree.HoeffdingAdaptiveTreeClassifier`. Ten se **doucuje po kazde
   svicce**, jakmile je znamy jeji skutecny vysledek (cena o `N` svicek
   pozdeji), a interne pouziva ADWIN detekci concept driftu - kdyz se chovani
   trhu zmeni, model se sam prizpusobi, aniz by ho nekdo musel rucne
   preskolovat.
2. **Bandit nad strategiemi** (`trading_agent/strategy/bandit.py`) - epsilon-greedy
   multi-armed bandit, ktery se uci, KTEREMU zdroji signalu (ML model /
   trend-following / mean-reversion) v aktualnim rezimu trhu vice duverovat,
   na zaklade realneho realizovaneho PnL z uzavrenych obchodu.

Vysledek: cim dele bot bezi (a cim vic obchodu uzavre), tim vic dat ma model
i bandit k dispozici a tim by se mela zlepsovat kvalita rozhodovani. Stav
obou (`data/model_state.pkl`, `data/bandit_state.json`) se pravidelne uklada
na disk, takze se uceni prenasi i pres restart agenta.

## Architektura

```
trading_agent/
├── config.py              nacteni nastaveni z .env (pydantic-settings)
├── agent.py                hlavni orchestrator - AutonomousTradingAgent
├── broker/alpaca_client.py jedine misto, kde kod mluvi s Alpaca API (alpaca-py)
├── data/
│   ├── bar_buffer.py        rolling buffer poslednich N svicek na symbol
│   └── storage.py           SQLite: obchody, signaly, equity krivka, metriky
├── features/indicators.py  technicke indikatory + feature engineering
├── model/online_model.py   online (river) samoucici se klasifikator smeru
├── strategy/
│   ├── rules.py              trend-following a mean-reversion pravidla
│   ├── bandit.py              epsilon-greedy bandit nad strategiemi
│   └── signal_engine.py       spoji model+pravidla+bandit do 1 signalu
├── risk/risk_manager.py    position sizing, stop/take-profit, limity
├── execution/order_manager.py  bracket ordery, reconciliace s Alpaca
├── backtest/backtester.py  event-driven walk-forward backtest
├── events.py               in-process pub/sub (agent -> dashboard)
└── webapp/
    ├── server.py            FastAPI: REST API + WebSocket, bearer-token auth
    ├── supervisor.py        start/stop/restart agenta z dashboardu
    ├── settings_api.py      popis nastaveni pro formular (hot vs. restart)
    └── static/index.html    dashboard (jedna stranka, bez build kroku)

scripts/
├── run_web.py        agent + webovy dashboard (doporucene spousteni)
├── run_live.py       jen agent, bez webu (headless provoz)
├── run_backtest.py   backtest nad historickymi daty
├── bootstrap_train.py "zahrivaci" beh - predtrenuje model/bandit pred prvnim startem
└── suggest_symbols.py vybere symboly, ktere se vejdou do kapitalu na uctu
```

### Tok dat za behu

1. Pri startu se stahne historie (`AlpacaBroker.get_historical_bars`) a naseje
   se do rolling bufferu (`MultiSymbolBarStore`).
2. Agent se pripoji na `StockDataStream` (websocket) a odebira nove svicky
   pro vsechny symboly z `SYMBOLS`.
3. Na kazdou novou svicku (`agent._on_bar`):
   - dopocitaji se indikatory (`build_feature_row`),
   - vyresi se "label" vzorku, kteremu prave uplynul predikcni horizont, a
     model se na nem doucí (self-learning krok),
   - `SignalEngine` vygeneruje signal (smer + jistota + pouzita strategie),
   - pokud neni jiz otevrena pozice a signal presahne `MIN_CONFIDENCE`,
     `RiskManager` spocita velikost pozice a stop/take-profit z ATR,
   - `OrderManager` posle bracket order (trh + SL + TP) na Alpaca.
4. Hlavni vlakno bezi nezavisle periodickou udrzbu: casove exity, detekci
   pozic uzavrenych primo Alpaca (SL/TP) a zpetnou vazbu do banditu, a
   periodicke ukladani stavu na disk.
5. Kazdy podstatny krok (signal, otevreni/uzavreni pozice, heartbeat, zmena
   stavu agenta) se publikuje do `EventBus` a websocketem tece rovnou do
   dashboardu.

## Webove rozhrani (dashboard)

```bash
python scripts/run_web.py          # agent + dashboard na http://127.0.0.1:8000/
```

Pri startu se do konzole vypise **pristupovy token** - ten se zadava pri prvnim
otevreni dashboardu (ulozi se do prohlizece). Agent i webovy server bezi ve
**stejnem procesu**, takze dashboard zobrazuje presne ten stav, se kterym agent
prave pracuje, a zmeny nastaveni se propisou okamzite bez mezikroku.

Co dashboard umi:

- **Zive sledovani** - stav agenta, equity/hotovost/kupni sila, graf vyvoje
  equity (vcetne tabulkoveho zobrazeni), otevrene pozice s aktualnim P/L,
  historie uzavrenych obchodu a signalu, kvalita modelu (rolling ROC AUC) a
  naucene vahy jednotlivych strategii z banditu.
- **Zivy prubeh** - websocketovy proud udalosti (signaly, otevrene/uzavrene
  pozice, heartbeaty) tak, jak je agent generuje.
- **Kompletni nastaveni** - vsechny parametry z `.env` jde upravit primo v
  rozhrani. Risk limity, `MIN_CONFIDENCE`, `DRY_RUN` apod. se v bezicim
  agentovi projevi **okamzite**; polozky oznacene `RESTART` (API klice,
  symboly, timeframe, predikcni horizont, data feed) vyzaduji tlacitko
  *Ulozit a restartovat agenta*. Vse se uklada do `data/runtime_settings.json`
  a prezije restart procesu (soubor je v `.gitignore`, stejne jako `.env`).
- **Ovladani** - start / stop / restart agenta, prepnuti kill-switche, zavreni
  jedne nebo vsech pozic (destruktivni akce maji potvrzovaci dialog).

### Bezpecnost dashboardu

Rozhrani umi menit risk limity a zavirat pozice na skutecnem uctu, proto:

- **Vychozi vazba je `127.0.0.1`** (`WEB_HOST`) - zvenci nedostupne.
- **Vsechny `/api/*` endpointy i websocket vyzaduji token** (`WEB_API_TOKEN`).
  Kdyz je prazdny, vygeneruje se pri prvnim startu nahodny a ulozi se.
- Pokud dashboard vystavite mimo localhost, postavte pred nej **HTTPS reverse
  proxy** a token drzte v tajnosti (skript na to pri startu upozorni).

## Bezpecnostni mechanismy

- **Paper trading jako vychozi** (`ALPACA_PAPER=true`) - na live ucet je
  potreba navic explicitne nastavit `I_UNDERSTAND_LIVE_TRADING_RISK=true`,
  jinak `scripts/run_live.py` odmitne start.
- **DRY_RUN** rezim - nic se neposila na Alpaca, jen se loguje a zaznamenava,
  co by bot udelal.
- **Kill-switch soubor** (`KILL_SWITCH_FILE`, vychozi `data/STOP`) - pouha
  existence souboru okamzite pozastavi nove vstupy (existujici pozice
  zustavaji chranene svymi bracket ordery).
- **Denni ztratovy limit** (`DAILY_LOSS_LIMIT_PCT`) a **drawdown circuit
  breaker** (`MAX_DRAWDOWN_PCT`) - po prekroceni bot prestane otevirat nove
  pozice (existujici pozice dal hlida jejich vlastni SL/TP na burze).
- **ATR-based risk sizing** - kazdy obchod riskuje jen `RISK_PER_TRADE` %
  equity, s hornim stropem `MAX_POSITION_PCT` % na jeden symbol.
- **Bracket ordery** - kazdy vstup ma stop-loss i take-profit nastaveny primo
  na burze, takze pozice je chranena, i kdyby proces bota spadl.
- Vsechny limity jsou nastavitelne v `.env` (viz `.env.example`).

## Instalace

Vyzaduje Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# a doplnit ALPACA_API_KEY / ALPACA_SECRET_KEY z app.alpaca.markets
# (paper i live klice se generuji samostatne)
```

## Pouziti

```bash
# 1) (volitelne, ale doporucene) over strategii na historickych datech
python scripts/run_backtest.py --days 90

# 2) (volitelne) "zahrej" model/bandit na historii pred prvnim ostrym behem,
#    aby agent nezacinal uplne od nuly
python scripts/bootstrap_train.py --days 90

# 3) spustit agenta s webovym dashboardem (doporucene)
python scripts/run_web.py
#    ...nebo bez webu, ciste headless:
python scripts/run_live.py
```

Zastaveni: `Ctrl+C` - agent se bezpecne odpoji od streamu a ulozi stav modelu
i banditu na disk (existujici otevrene pozice zustavaji beze zmeny, pokud
neni `CLOSE_POSITIONS_ON_SHUTDOWN=true`).

## Konfigurace

Vsechny parametry jsou v `.env` (vzor v `.env.example`), mimo jine:

| Promenna | Vyznam |
|---|---|
| `SYMBOLS` | sledovane symboly, oddelene carkou |
| `TIMEFRAME_MINUTES` | delka jedne svicky v minutach |
| `PREDICTION_HORIZON_BARS` | za kolik svicek dopredu model predikuje smer |
| `RISK_PER_TRADE` | podil equity riskovany na jeden obchod |
| `MAX_POSITION_PCT` | max. podil equity v jednom symbolu |
| `MAX_OPEN_POSITIONS` | max. soucasne otevrenych pozic |
| `DAILY_LOSS_LIMIT_PCT` / `MAX_DRAWDOWN_PCT` | denni a celkovy risk limit |
| `MIN_CONFIDENCE` | minimalni jistota signalu pro vstup |
| `ALLOW_SHORT` | povolit short prodeje |
| `DRY_RUN` | simulace bez skutecneho odesilani orderu |
| `WEB_HOST` / `WEB_PORT` | adresa a port dashboardu |
| `WEB_API_TOKEN` | pristupovy token do dashboardu (prazdny = vygeneruje se) |

Vetsinu z nich lze menit i za behu primo v dashboardu (zalozka *Nastaveni*).

### Odkud se nastaveni bere (dve vrstvy)

1. **`.env`** - zaklad, ktery plati vzdy.
2. **`data/runtime_settings.json`** - jen ty polozky, ktere jste **explicitne
   ulozili v dashboardu** (plus automaticky vygenerovany `WEB_API_TOKEN`).
   Tyto hodnoty maji prednost pred `.env`.

Zamerne se tedy neuklada kompletni kopie nastaveni - jinak by prvni spusteni
zmrazilo tehdejsi obsah `.env` (vcetne API klicu) a pozdejsi oprava `.env` by
se uz neprojevila. Co prave prebiji `.env`, vypise agent pri startu do logu;
smazanim `data/runtime_settings.json` se vratite plne k `.env` (vygeneruje se
novy pristupovy token).

## Maly ucet (stovky dolaru)

Bracket ordery na Alpaca neumi zlomkove akcie, takze velikost pozice se vzdy
zaokrouhluje dolu na cele kusy. Z toho plyne tvrde pravidlo:

> **1 cely kus musi stat max `equity * MAX_POSITION_PCT`.**

Pri kapitalu 230 USD a vychozim `MAX_POSITION_PCT=0.2` je strop na jednu pozici
46 USD - drahe tituly (SPY, MSFT, AAPL) by tedy neotevrely **zadnou** pozici.
Agent na to upozorni hned po startu i pri kazdem zablokovanem vstupu, ale
symboly je potreba vybrat podle kapitalu:

```bash
python scripts/suggest_symbols.py                 # podle skutecne equity z uctu
python scripts/suggest_symbols.py --equity 230    # nebo pro konkretni castku
```

Skript stahne aktualni ceny z Alpaca, spocita kolik kusu se vejde do limitu a
vypise hotovy radek `SYMBOLS=` k vlozeni do `.env` nebo do dashboardu. Po zmene
symbolu je potreba restartovat agenta (tlacitko v dashboardu).

Dalsi dve veci, ktere u malych uctu plati bez ohledu na tento software:

- **Typ uctu.** Margin ucet vyzaduje u brokera min. 2 000 USD, takze ucet za
  stovky dolaru je zpravidla **cash**. Na cash uctu neplati PDT limit (max. 3
  day trades za 5 dni), zato se penize z prodeje zuctovavaji az druhy pracovni
  den (T+1) - realne tak zvladnete radove jeden obchodni cyklus denne z plne
  castky. Typ uctu si overte primo v Alpaca dashboardu.
- **Backtest s realnou castkou.** `python scripts/run_backtest.py --equity 230`,
  jinak backtest pocita s vychozimi 100 000 USD a vysledky nebudou porovnatelne.

## Reseni problemu

**`APIError: {"message": "unauthorized."}` / HTTP 401 na `/v2/account`**

Alpaca odmitla pristupove udaje. Agent to od verze s dashboardem hlasi
konkretnim navodem misto tracebacku; projdete postupne:

1. **Paper a live ucet maji ODLISNE klice.** Pri `ALPACA_PAPER=true` musite
   pouzit klice vygenerovane pro *paper* ucet na app.alpaca.markets (paper
   klice zpravidla zacinaji `PK`, live `AK`). Nejcastejsi pricina.
2. Key i secret jsou zkopirovane cele, bez mezer a uvozovek. Secret Alpaca
   zobrazi jen jednou pri vytvoreni - pokud ho nemate, vygenerujte novy par.
3. Klice nebyly mezitim na Alpaca regenerovane nebo smazane.
4. Zkontrolujte `data/runtime_settings.json` - pokud jste klice zadavali v
   dashboardu, maji prednost pred `.env`. Opravte je tamtez, nebo soubor smazte.

**Agent nebezi, dashboard ale funguje** - podivejte se na `last_error` v hlavicce
dashboardu (a do `data/logs/trading_agent.log`); po oprave nastaveni staci
kliknout na *Spustit agenta*, restart procesu neni potreba.

## Testovani

```bash
pytest -v
```

Testy pokryvaji vsechnu cistou logiku bez potreby sitoveho pripojeni k Alpaca
(indikatory, risk management, bandit, online model, pravidlove strategie,
storage, order manager s fake brokerem, backtester, a webove API vcetne
autentizace, validace nastaveni a websocketu). Kod komunikujici primo s Alpaca
API (`broker/alpaca_client.py`, plny beh `agent.run()`) neni pokryt
automatizovanymi testy, protoze vyzaduje skutecne API klice a sitovy pristup -
overte ho rucne pres `scripts/run_backtest.py` (stahuje realna historicka data)
a nasledne kratkym behem na paper uctu.

## Zname limity

- Backtester prehrava indikatory bar-po-baru a pro kazdou svicku prepocita
  cely feature set nad rolling oknem - u velmi dlouhych obdobi (stovky dni)
  proto muze beh trvat radove minuty.
- Datovy feed je ve vychozim nastaveni IEX (`DATA_FEED=iex`), coz je feed
  dostupny i bez placene Alpaca SIP subscription; pro presnejsi data lze
  prepnout na `sip`, pokud ma ucet prislusne opravneni.
- Pending vzorky cekajici na "label" (posledních `PREDICTION_HORIZON_BARS`
  svicek na symbol) se pri restartu agenta nezachovavaji - jde o zanedbatelne
  mnozstvi dat v radu minut, ktere se rychle nahradi novymi.
