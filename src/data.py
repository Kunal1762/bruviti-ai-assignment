"""Synthetic complaint generator for Bruviti Use Case 1 (fault classification).

Run from the repo root:   python -m src.data
Writes:                   data/synthetic_complaints.csv

Design (details go in DATA.md):
* The unit of data is a SCENARIO: one underlying situation (e.g. "washer drum
  does not spin after filling") with 2-4 genuinely different paraphrases.
* Scenarios, not rows, are assigned to train / dev / test. Every row of a
  held-out scenario is held out, so evaluation measures unseen situations
  and unseen wordings, not memorised templates.
* Each row = scenario paraphrase x product noun x wrapper (opener, irrelevant
  detail, closer, sometimes a long narrative) x noise (typos, case, text-speak).
* Product-aware: a scenario lists the products it is valid for, so there are
  no fridge-with-drum-fault rows.
* Ambiguous scenarios are labelled by C5 (the earlier label in Appendix B).
* Safety rules C1/C2 are NOT enforced here; they belong to the rule engine.
  A few spark/smoke rows exist only so the ML model has seen the vocabulary.
* All text is written by the author of this file, so it shares one author's
  style. Synthetic scores must not be read as hidden-test performance.
"""
from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261013
ROWS_PER_LABEL = 144  # target; dedup may drop a few
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "synthetic_complaints.csv"
CSV_FIELDS = ["id", "product_category", "complaint_text", "fault_category",
              "split", "scenario_id", "kind"]

# Order matters: C5 tie-breaking uses this order.
LABELS = [
    "no_power", "not_heating", "not_cooling", "water_leak",
    "drainage_or_blockage", "noise_or_vibration", "door_seal_or_latch",
    "control_panel_or_display", "electrical_or_control_board",
    "motor_or_drive", "installation_or_setup", "other",
]
PRODUCTS = [
    "washing_machine", "refrigerator", "dishwasher", "oven_or_cooktop",
    "air_conditioner", "water_heater", "microwave",
]
_P = dict(W="washing_machine", F="refrigerator", D="dishwasher", O="oven_or_cooktop",
          A="air_conditioner", H="water_heater", M="microwave")
ALL = "WFDOAHM"
VALID = {label: ALL for label in LABELS}
VALID["not_heating"] = "WDOAHM"   # heating products (AC heat mode included)
VALID["not_cooling"] = "FA"       # cooling products
VALID["drainage_or_blockage"] = "WDFA"

NOUNS = {
    "W": ["washing machine", "washer", "front loader", "washer dryer"],
    "F": ["fridge", "refrigerator", "fridge freezer", "freezer"],
    "D": ["dishwasher", "dish washer"],
    "O": ["oven", "cooker", "cooktop", "hob", "range"],
    "A": ["AC", "air conditioner", "split AC", "window AC"],
    "H": ["water heater", "geyser", "heater"],
    "M": ["microwave", "microwave oven"],
}
CODES = ["E21", "E1", "F4", "H20", "E24", "F21", "E15", "E8", "U4", "d90",
         "1E", "E3", "F0", "H1", "E10", "Err 3", "E-05", "E33", "F9", "E40"]


def sc(kind, prods, *paras):
    """kind: plain|indirect|ambiguous|code|overlap|multilingual"""
    return (kind, prods, list(paras))


# --------------------------------------------------------------------------
# Scenario bank. {P} = product noun, {C} = error code. Ambiguous scenarios sit
# under the label C5 picks (the earlier one in Appendix B).
# --------------------------------------------------------------------------
SCENARIOS = {
"no_power": [
    sc("plain", ALL, "{P} won't switch on at all, tried a different socket",
       "Completely dead. Nothing lights up when I press the power button.",
       "hi, my {P} has no power, it just doesn't come on since this morning"),
    sc("plain", ALL, "There are no lights, no beeps, no sound at all when I plug it in",
       "I plugged in the {P} and nothing happens. The wall outlet works for my phone charger.",
       "unit unresponsive, will not power up"),
    sc("plain", ALL, "after the power cut yesterday it never came back on",
       "Power went out in our area for a few hours and since then the {P} has stayed off",
       "stopped working overnight, dead this morning, outlet is fine"),
    sc("plain", "W", "washer completely lifeless, no lights on the console, tried resetting at the wall",
       "my washing machine is dead. Pressing start does nothing and the panel is totally dark",
       "the machine doesn't turn on, not even the little light"),
    sc("plain", "F", "fridge light is off and there's no hum, completely dead, food still cold for now",
       "no power to my fridge freezer at all, interior light not working either",
       "fridge seems to have died, nothing running at all"),
    sc("plain", "D", "dishwasher is totally dead, press the button and no response, no lights",
       "Dishwasher no longer powers on. The fuse in the plug is good.",
       "doesn't start, no sign of life from the dishwasher"),
    sc("plain", "OM", "{P} is dead, no clock, no lights, nothing at all",
       "The {P} won't turn on. Display is off and nothing responds.",
       "completely without power, tried another socket, still nothing"),
    sc("plain", "A", "AC won't turn on, no lights on the indoor unit and the remote has no effect",
       "my air conditioner is totally dead, no LED, no response from remote or unit",
       "split AC has no power at all, tried the switch on the wall too"),
    sc("plain", "H", "geyser switch is on but the indicator never lights up, completely dead",
       "the water heater has no power, no glow from the light, nothing",
       "geyser is not coming on at all, no light on the unit"),
    sc("ambiguous", ALL, "screen is black and none of the buttons work, I think it has no power or the panel is broken",
       "No lights, no response from any button - either dead or the control panel has gone",
       "can't turn it on, display not lighting up, not sure if it's the power or the buttons"),
    sc("plain", ALL, "It won't power on. I've checked the plug, the socket and the fuse box and all are fine.",
       "Does not switch on. Same outlet works with other appliances.",
       "brand new but dead as soon as I plugged it in, no response"),
    sc("indirect", ALL, "I pressed the button and waited, then pressed again, then tried again, nothing. Zero response.",
       "Tried everything I could think of to get it going and it just sits there silent and dark",
       "it doesn't start"),
    sc("multilingual", ALL, "{P} on hi nahi ho rahi, bilkul dead hai, light bhi nahi aa rahi",
       "mi {P} no enciende, esta muerto, sin luz",
       "ca ne s'allume plus du tout, aucun voyant"),
],
"not_heating": [
    sc("plain", "WDOAHM", "{P} isn't heating up properly",
       "No heat coming from it at all",
       "it takes ages to get hot and never reaches the temperature I set"),
    sc("plain", "W", "washer doesn't heat the water, clothes come out as if washed cold even on 60 degrees",
       "Hot wash is just cold water in my washing machine, heating element must have failed",
       "wash water stays freezing whatever temperature I select"),
    sc("plain", "D", "dishes come out cold and greasy because the water never heats up",
       "dishwasher doesn't dry the dishes at all and the inside stays cold after the cycle",
       "Rinse water is cold and the detergent hasn't dissolved, heating seems dead"),
    sc("plain", "O", "oven stays cold even after 20 minutes of preheating",
       "I set 200 degrees and it only reaches about 100, baking comes out raw",
       "Preheat beeped but the cavity is barely warm"),
    sc("plain", "O", "hob ring stays cold when I turn it on, only the other rings get hot",
       "gas burner won't light and I can smell gas, no flame at all",
       "the electric plate on the cooktop takes forever and never gets properly hot"),
    sc("plain", "H", "geyser gives only cold water, no hot water at all",
       "water heater is lukewarm and runs out of hot water in two minutes",
       "Took a shower and it was freezing, the heater switch is on"),
    sc("plain", "H", "Instant heater water is barely warm even on the highest setting",
       "hot water takes forever to arrive and then it's only tepid",
       "heater light is on but the water that comes out is cold"),
    sc("plain", "M", "microwave runs and the light is on but the food stays stone cold",
       "Put soup in for 3 minutes, still cold. It's not heating at all.",
       "microwave sounds like it's working but doesn't warm anything"),
    sc("plain", "A", "AC in heating mode only blows cold air",
       "heat mode on my split AC doesn't produce any warm air in winter",
       "in winter the AC just blows room temperature air even in heat mode"),
    sc("ambiguous", "WOA", "{P} powers up and the fan or motor runs but there is no heat, can't tell if the element or the motor is at fault",
       "it's running but not heating, either the heating part or the motor is gone",
       "runs fine, sounds fine, but no heat at all"),
    sc("indirect", "OM", "food is still cold in the middle after the full cooking time, edges barely warm",
       "Dinner came out raw and lukewarm again, same time and setting as always",
       "cooking times have doubled and it still isn't hot enough"),
    sc("multilingual", "WDOAHM", "{P} garam nahi ho raha, bahut der se try kar rahe hain",
       "{P} no calienta nada, esta frio",
       "ca ne chauffe plus du tout, c'est froid"),
],
"not_cooling": [
    sc("plain", "F", "fridge is warm inside and the milk goes off in a day",
       "everything in my fridge is lukewarm, vegetables are wilting",
       "The fridge isn't getting cold anymore, thermometer says 14 degrees"),
    sc("plain", "F", "freezer is thawing, ice cream melted and the meat is soft",
       "freezer won't freeze anything any more",
       "Everything in the freezer section has defrosted overnight"),
    sc("plain", "F", "thick frost has built up on the back wall of the freezer and the fridge isn't cold",
       "ice keeps forming around the freezer door and cooling is weak",
       "Ice build-up everywhere in the freezer compartment, never used to do this"),
    sc("plain", "A", "AC runs all day but the room doesn't get cool",
       "air from the AC is just warm, no cooling at all",
       "The AC is set to 18 degrees and blows room temperature air"),
    sc("plain", "A", "ice forming on the indoor unit pipe and cooling has dropped a lot",
       "AC blowing weak air that isn't cold, the coil seems frozen over",
       "pipe is covered in ice, hardly any cool air"),
    sc("plain", "A", "cooling is very poor since last week, the room takes hours to cool",
       "the AC used to cool in 15 minutes and now it can't get below 28",
       "My bedroom AC just doesn't cool like before"),
    sc("plain", "FA", "not cold enough, the temperature keeps rising",
       "Cooling performance has dropped badly, nowhere near cold enough",
       "it's running but there's no chill at all"),
    sc("indirect", "F", "opened my fridge and the butter is soft and the ice trays are slush",
       "ice cubes are all stuck together and drinks aren't chilled",
       "Yogurt has gone off even though I bought it two days ago and the inside feels warm"),
    sc("indirect", "A", "it feels like the AC is just a fan now",
       "just blowing air, nothing cold about it",
       "the room is hotter than outside with the AC on"),
    sc("ambiguous", "FA", "{P} is not cooling and I don't think the compressor is starting, it's warm",
       "not cooling, compressor seems not to start",
       "warm inside and I can't hear the compressor kick in"),
    sc("ambiguous", "FA", "{P} isn't cold and there's some water on the floor underneath as well",
       "not cooling and a bit of water is leaking out too",
       "no cooling, plus a wet patch next to it"),
    sc("multilingual", "FA", "{P} thanda nahi kar raha, bilkul cooling nahi hai",
       "{P} no enfria nada, la comida se esta echando a perder",
       "ca ne refroidit plus du tout"),
],
"water_leak": [
    sc("plain", ALL, "there's a puddle of water under my {P}",
       "water all over the floor around the {P} this morning",
       "found the floor wet next to it, it must be leaking"),
    sc("plain", "W", "washing machine leaks from the bottom during the wash, floor flooded",
       "water gushing out underneath the washer in the middle of the spin",
       "detergent drawer overflows and water runs down the front"),
    sc("plain", "W", "water pours from the bottom of the washer after every fill",
       "leaking from the hose connection at the back of the machine",
       "pool of water in front of the washer after every wash"),
    sc("plain", "F", "water collecting at the bottom of the fridge and on the floor",
       "puddle under the fridge every morning",
       "fridge leaking water inside onto the shelves from the back panel"),
    sc("plain", "D", "dishwasher leaks out the front onto the floor every cycle",
       "water seeping from the bottom of the dishwasher door",
       "Soapy water on the kitchen floor after each wash"),
    sc("plain", "A", "water dripping from the indoor unit onto the floor",
       "AC is leaking water down the wall",
       "my AC drips continuously, there's a wet patch on the carpet"),
    sc("plain", "H", "geyser tank leaking from the bottom",
       "water dripping from the pressure valve pipe constantly",
       "The water heater leaks at the pipe joints and the floor is soaked"),
    sc("plain", "OM", "water dripping out from the bottom of the {P}",
       "found a small pool of water under the {P}, no idea where it comes from",
       "water has leaked onto the counter from the {P}"),
    sc("ambiguous", "WDFA", "water won't drain and it's also overflowing onto the floor",
       "blocked drain and water is spilling all over the floor",
       "it doesn't empty properly and water leaks out of the front"),
    sc("ambiguous", "WD", "rubber seal is torn and water leaks out of the door",
       "water drips from the door gasket where it's split",
       "leaking from around the door, the seal looks damaged"),
    sc("indirect", ALL, "my neighbour downstairs complained about water coming through their ceiling",
       "I keep mopping the floor every morning, it's soaked again",
       "The cabinet next to it has swollen from the wet"),
    sc("plain", "WDH", "overflowed while filling and flooded the room",
       "it filled and kept filling until it spilled over the top",
       "water level rose and overflowed, whole kitchen under water"),
    sc("multilingual", ALL, "{P} ke neeche se paani leak ho raha hai, floor geela hai",
       "hay agua en el suelo debajo del {P}",
       "ca fuit sous le {P}, il y a de l'eau par terre"),
],
"drainage_or_blockage": [
    sc("plain", "W", "water stuck in the drum after the cycle, won't drain",
       "the machine doesn't empty, clothes are sitting in dirty water",
       "stops at the drain stage and the tub stays full"),
    sc("plain", "W", "filter was clogged with coins and fluff, water won't pump out",
       "drain hose seems blocked, water just won't go out",
       "cleaned the filter and found it full of gunk but it still drains very slowly"),
    sc("plain", "D", "dirty water sits at the bottom of the dishwasher after each wash",
       "dishwasher doesn't drain, standing water with food bits",
       "water remains in the tub and smells, the filter looks blocked"),
    sc("plain", "D", "The sump is clogged and the waste water won't leave",
       "sink backs up into the dishwasher when it runs",
       "dishwasher drains extremely slowly, the filter is full of food"),
    sc("plain", "F", "defrost drain is blocked and water collects in the bottom of the fridge",
       "ice melt water can't drain from the back and sits in the tray",
       "the little drain hole at the back of the fridge is clogged, standing water inside"),
    sc("plain", "A", "AC drain pipe is blocked, condensate isn't draining out",
       "the drain pipe of my split AC is choked with dirt and the water tray stays full",
       "condensate not going out through the pipe, looks clogged"),
    sc("plain", "WD", "pump seems blocked, I can hear it trying but no water leaves",
       "drain pump is jammed with debris and nothing flows out",
       "can't get the water out, I think the pump filter is blocked"),
    sc("indirect", "WD", "after the wash the clothes are soaking and heavy, tub still half full of grey water",
       "there's still water in it hours after the programme finished",
       "dishes still sitting in a pool of dirty water when I opened it"),
    sc("plain", "WDFA", "water is not draining out",
       "blocked filter, nothing goes down",
       "it won't empty"),
    sc("ambiguous", "WD", "drain pump hums but no water leaves, the pump might be dead or blocked",
       "water isn't draining and the pump seems not to run, blocked or broken?",
       "not draining, not sure whether the pump is jammed or has failed"),
    sc("ambiguous", "WD", "it won't drain and makes a gurgling noise when it tries",
       "gurgling sounds from the drain and the water stays in the tub",
       "grinding and no draining"),
    sc("multilingual", "WD", "paani drain nahi ho raha, tub mein bhara hua hai",
       "el agua no se va, el filtro esta tapado",
       "l'eau ne s'evacue pas, le filtre est bouche"),
],
"noise_or_vibration": [
    sc("plain", "W", "washing machine bangs against the wall during spin and walks across the floor",
       "Drum thuds and the whole machine shakes violently at high spin",
       "excessive vibration in the spin cycle, it moves about the room"),
    sc("plain", "W", "loud grinding noise from the drum when washing",
       "screeching sound while the drum turns",
       "there's a rumbling like a jet engine during the final spin"),
    sc("plain", "F", "fridge makes a loud buzzing noise at night",
       "rattling and humming from the back of the fridge all the time",
       "constant clicking noise from my fridge every few minutes"),
    sc("plain", "D", "dishwasher is really loud, grinding sound during the wash",
       "banging noise when the spray arm moves",
       "my dishwasher screeches and rattles, much noisier than before"),
    sc("plain", "O", "oven makes loud popping and cracking sounds when heating up",
       "cooktop clicks continuously",
       "oven buzzes loudly while it's on"),
    sc("plain", "A", "outdoor unit rattles and vibrates loudly",
       "AC makes a whistling noise from the indoor unit",
       "loud banging from the outside unit whenever it starts"),
    sc("plain", "H", "geyser makes gurgling and banging sounds when heating",
       "water heater rumbles and pops like popcorn",
       "hissing and cracking from the tank"),
    sc("plain", "M", "microwave makes a loud humming and buzzing noise while running",
       "grinding sound coming from the microwave",
       "it screams while it runs"),
    sc("plain", ALL, "strange noise coming from it, never did that before",
       "very loud since last week, my family can hear it from the next room",
       "weird knocking sound whenever it's running"),
    sc("indirect", ALL, "it's so loud I can't hear the TV in the other room",
       "keeps me awake at night, a sort of droning",
       "The noise has got worse over a month"),
    sc("ambiguous", ALL, "bearing sounds bad, grinding noise from inside the {P}, it still runs though",
       "fan rattles and squeals but is running",
       "loud noise from the motor, it still turns"),
    sc("multilingual", ALL, "bahut zor ki awaaz aa rahi hai, {P} hil rahi hai",
       "hace un ruido muy fuerte y vibra mucho",
       "ca fait un bruit enorme et ca vibre"),
],
"door_seal_or_latch": [
    sc("plain", "WFDOM", "the door won't close properly, there's a gap",
       "door is stuck and I can't get it open",
       "door handle snapped off in my hand"),
    sc("plain", "W", "door lock is stuck after the cycle, can't open it to take clothes out",
       "the door won't unlock, light says locked but the wash is finished",
       "plastic door catch on my washer has broken, door keeps popping open"),
    sc("plain", "W", "rubber door gasket is torn and has mould",
       "seal around the door is split",
       "front loader rubber seal is perished and falling apart"),
    sc("plain", "F", "fridge door doesn't seal, the rubber is loose and the door pops open",
       "gasket on the freezer door is worn and the door doesn't stay shut",
       "the door of the fridge swings open by itself, the magnet strip seems gone"),
    sc("plain", "D", "dishwasher door won't latch so the cycle refuses to start",
       "door catch is broken and it doesn't click shut",
       "door hangs loose and won't stay closed"),
    sc("plain", "O", "oven door won't close fully, the hinge is broken",
       "hinge on the oven door has failed and it droops open",
       "oven door latch snapped and it won't lock"),
    sc("plain", "M", "microwave door won't latch, the catch is broken",
       "door on the microwave doesn't close and keeps popping open",
       "microwave door handle came off and it won't shut properly"),
    sc("plain", "A", "front panel of the AC won't latch and hangs open",
       "AC flap is stuck open and won't close",
       "the cover on the indoor unit pops open"),
    sc("indirect", "WFDOM", "there's a gap along the edge of the door when it's shut",
       "the door doesn't sit flush anymore",
       "I have to slam the door twice for it to catch"),
    sc("ambiguous", "WDOM", "panel says door open but the door is shut, the latch might be faulty",
       "door error on the display, the latch seems not to catch",
       "Display keeps showing door open though it's closed"),
    sc("plain", "WDM", "won't start because the door doesn't latch",
       "it refuses to start, the door switch seems broken",
       "cycle won't begin, door doesn't lock shut"),
    sc("multilingual", "WFDOM", "darwaza band nahi ho raha, latch toot gaya hai",
       "la puerta no cierra bien, la junta esta rota",
       "la porte ne ferme plus, le joint est abime"),
],
"control_panel_or_display": [
    sc("plain", ALL, "buttons don't respond when I press them",
       "touch panel is not reacting at all",
       "keypad does nothing, just a beep sometimes"),
    sc("plain", ALL, "display is blank but the unit seems to be on",
       "screen shows garbled characters and strange symbols",
       "the display is flickering and unreadable"),
    sc("plain", ALL, "settings keep resetting by themselves",
       "it forgets my settings every time, the clock goes back to 12:00",
       "I set the timer and it gets cleared immediately"),
    sc("plain", "W", "programme dial doesn't change the cycle and the display is garbled",
       "can't select any wash programme, buttons are dead",
       "the start button doesn't respond and the screen shows nonsense"),
    sc("plain", "F", "fridge temperature display is blank and the buttons don't work",
       "can't change the temperature setting, the panel doesn't respond",
       "digital display on the fridge door shows wrong numbers and won't change"),
    sc("plain", "D", "touch buttons on the dishwasher are dead, can't pick a programme",
       "display shows random characters and I can't select anything",
       "the programme lights flash but no button works"),
    sc("plain", "OM", "clock display on the {P} is blank and the timer buttons won't set",
       "touch controls on the {P} unresponsive, only the light works",
       "{P} keypad doesn't register presses and the display is scrambled"),
    sc("plain", "A", "AC remote doesn't control the unit, the display shows the wrong temperature",
       "buttons on the indoor unit don't work and the remote is ignored",
       "temperature display stuck on one number and the mode won't change"),
    sc("plain", "H", "digital display of the geyser is blank, can't set the temperature",
       "temperature keys don't respond, screen is dead",
       "geyser panel shows garbage and the buttons don't work"),
    sc("code", ALL, "display shows {C}",
       "it just says {C} and nothing else",
       "error code {C} on the screen, no idea what it means"),
    sc("code", ALL, "getting {C} flashing on the panel",
       "Fault {C} displayed",
       "what does {C} mean on my {P}?"),
    sc("ambiguous", ALL, "display flickers and the buttons only work sometimes, maybe the board",
       "screen goes blank at random and the controls act up, board problem?",
       "panel is intermittent, the display dies and comes back"),
    sc("multilingual", ALL, "buttons kaam nahi kar rahe, display band hai",
       "el panel no responde y la pantalla esta en blanco",
       "l'ecran est vide et les boutons ne repondent plus"),
        sc("code", ALL, "{C} keeps coming up on the display",
       "Panel reads {C}",
       "I see {C} on the screen, what does it mean"),
    sc("code", ALL, "it's showing error {C}",
       "got a {C} message on the display",
       "error {C}, please advise"),
],
"electrical_or_control_board": [
    sc("plain", ALL, "there was a spark from the back when I switched it on",
       "saw sparks near the plug and switched everything off",
       "small flash and a bang from behind it, then nothing"),
    sc("plain", ALL, "burning smell coming from the {P}",
       "smells of burnt plastic when it runs",
       "a strong acrid smell like something is burning, I unplugged it"),
    sc("plain", ALL, "smoke came out of the {P} and I pulled the plug",
       "wisp of smoke from the vents when I turned it on",
       "grey smoke from the back panel"),
    sc("plain", ALL, "got an electric shock when I touched the metal body",
       "tingling shock from the {P} when I touch it with wet hands",
       "mild shock from the casing, maybe an earth problem"),
    sc("plain", ALL, "it trips the MCB every time I switch it on",
       "the whole house power trips as soon as the {P} starts",
       "blows the fuse each time I use it"),
    sc("plain", ALL, "it keeps turning off and on by itself, the power is intermittent",
       "randomly cuts out then restarts on its own",
       "works for ten minutes and then dies, then comes back later"),
    sc("plain", ALL, "control board seems faulty, scorch marks visible on the PCB",
       "main board burnt, the technician said it needs a new PCB",
       "after a voltage surge the board got fried and it acts erratically"),
    sc("plain", "W", "washer restarts itself mid-cycle and the display resets",
       "machine randomly shuts down in the middle of washing then comes back, board issue",
       "the washing machine's main board keeps rebooting"),
    sc("plain", "F", "fridge keeps switching itself off and on and the control board is clicking",
       "power to the fridge cuts out intermittently, relay clicking on the board",
       "fridge dies randomly and restarts after an hour"),
    sc("plain", "M", "sparks inside the microwave, flashes when it runs",
       "arcing and a burning smell inside the cavity",
       "blue light and crackling inside, smell of burning"),
    sc("overlap", ALL, "stopped working and there was a burning smell from the back, also the display is blank",
       "There was a spark, and now the buttons don't work either",
       "noisy, then a smell of burning and smoke, then it died"),
    sc("multilingual", ALL, "chingari nikli aur jalne ki smell aa rahi hai",
       "salio una chispa y huele a quemado",
       "ca sent le brule et il y a eu une etincelle"),
        sc("overlap", ALL, "very loud buzzing and then a burning smell, switched it off at the wall",
       "making a racket, then smoke, I unplugged it",
       "loud noise and sparks at the back"),
    sc("overlap", "WFDOM", "the door is stuck and I smelled burning from the control area",
       "door wouldn't close and then a spark near the hinge wiring",
       "door is jammed and there was smoke from the handle side"),
],
"motor_or_drive": [
    sc("plain", "W", "drum doesn't turn, it fills with water and then just sits there",
       "washing machine fills up, then the drum never rotates, shows {C}",
       "drum won't spin in the spin cycle, clothes come out soaking"),
    sc("plain", "W", "motor hums but the drum doesn't move",
       "belt seems to have snapped, the motor runs but nothing turns",
       "the agitator isn't moving during the wash"),
    sc("plain", "F", "fridge compressor clicks and doesn't start",
       "I can hear the compressor try to start then click off",
       "the compressor doesn't run at all but the light works"),
    sc("plain", "F", "freezer fan doesn't spin, I can see it's stuck",
       "evaporator fan isn't running",
       "condenser fan at the back never turns"),
    sc("plain", "D", "spray arm doesn't rotate so the dishes stay dirty",
       "circulation pump doesn't run, nothing is sprayed",
       "dishwasher fills but the wash motor doesn't start"),
    sc("plain", "O", "fan in the oven doesn't turn",
       "convection fan has stopped rotating",
       "the cooling fan of the oven keeps running long after switching off"),
    sc("plain", "A", "indoor blower fan isn't turning, no airflow",
       "AC compressor is not starting, the outdoor fan hums but doesn't spin",
       "outdoor fan has stopped turning"),
    sc("plain", "H", "circulation pump on my instant heater doesn't run",
       "booster pump not starting, no water flow through the heater",
       "the pump hums but there is no flow"),
    sc("plain", "M", "turntable doesn't rotate while it runs",
       "glass plate stays still and food heats unevenly",
       "plate won't spin, the motor underneath is silent"),
    sc("plain", ALL, "motor is running but nothing turns",
       "I can hear it trying but the moving part just doesn't move",
       "runs very slowly and jerky, doesn't reach full speed"),
    sc("ambiguous", ALL, "just set it up and the moving parts don't move at all",
       "brand new and the fan doesn't turn on first use",
       "first time running it and the motor does nothing"),
    sc("multilingual", ALL, "{P} ka motor hum karta hai lekin ghoomta nahi",
       "el motor zumba pero no gira",
       "le moteur ronronne mais rien ne tourne"),
],
"installation_or_setup": [
    sc("plain", "W", "new washer not level after fitting, it wobbles and shakes",
       "transit bolts still in, machine jumping around, just delivered",
       "installer connected the hoses but the inlet doesn't fit the tap"),
    sc("plain", "W", "can't connect the water inlet hose to our tap, wrong size",
       "drain hose doesn't reach the standpipe, don't know how to fit it",
       "need help installing and setting up the new washing machine, first use"),
    sc("plain", "F", "delivered yesterday, the doors are not level and one doesn't close properly",
       "new fridge rocks on its feet and the doors don't line up",
       "fridge is too wide for the gap in the kitchen, doesn't fit"),
    sc("plain", "F", "just got it, how long should I wait before switching on? first time setup",
       "new fridge delivered today, how do I set up the temperature and fit the shelves",
       "unsure how to connect the water line for the ice maker"),
    sc("plain", "D", "dishwasher not sitting flush with the cabinet after installation",
       "inlet and drain hose fittings don't match, newly delivered dishwasher",
       "the installer left it unconnected and I can't get it to work"),
    sc("plain", "O", "new cooker gas connection isn't fitted properly, the installer left",
       "oven doesn't fit into the cabinet cutout",
       "hob installed crooked and doesn't sit flat on the counter"),
    sc("plain", "A", "split AC installed but the pipe connection looks wrong and the outdoor unit is loose",
       "AC wall bracket isn't level and the unit looks crooked",
       "first time use, don't know how to set up the AC"),
    sc("plain", "H", "geyser installed with the inlet and outlet pipes reversed",
       "water heater wall mount isn't level after fitting",
       "newly fitted geyser, the plumber left connections loose and I don't know how to set it"),
    sc("plain", "M", "new microwave first use, no manual and I don't know how to set it up",
       "built-in microwave doesn't fit the trim kit",
       "microwave won't sit level on the shelf after delivery"),
    sc("plain", ALL, "just delivered and I can't get it up and running",
       "brand new out of the box and I'm struggling to set it up",
       "the fitter left it crooked and the connections look wrong"),
    sc("indirect", ALL, "it doesn't sit straight",
       "needs levelling, it wobbles on the floor",
       "isn't level and rocks when I push it"),
    sc("multilingual", ALL, "nayi {P} lagayi hai, level nahi hai, pipe connection samajh nahi aa raha",
       "acabo de instalar el {P} y no esta nivelado",
       "l'appareil vient d'etre installe, il n'est pas de niveau"),
],
"other": [
    sc("plain", ALL, "Thank you so much, the technician was polite and fixed everything quickly",
       "Just wanted to say thanks to your team, the {P} works great now",
       "great service, five stars, many thanks"),
    sc("plain", ALL, "thanks, the engineer fixed the leak yesterday and everything is fine now",
       "issue is resolved, the noise is gone after the repair, thank you",
       "problem solved after your visit, closing this complaint, thank you"),
    sc("plain", ALL, "I want to buy a new {P}, what models do you have and what are the prices",
       "Do you have any discount on your latest range? Looking to buy",
       "Can you send a quotation for 10 units for our hotel"),
    sc("plain", ALL, "what are your installation charges for a new split AC",
       "do you offer free installation with a new purchase",
       "How much do you charge to install and set up a new unit?"),
    sc("plain", ALL, "Congratulations! You have won a free gift card, click the link to claim",
       "Earn $500 a day working from home, no experience needed",
       "We can rank your website number 1 on Google, reply for a free audit"),
    sc("plain", ALL, "URGENT: your account is suspended, verify your password now",
       "Buy cheap watches online, limited offer",
       "Subscribe to my channel for free giveaways"),
    sc("plain", ALL, "what is the warranty period on my purchase",
       "please send me the user manual for my {P}",
       "where is your nearest service centre and what are the opening hours"),
    sc("plain", ALL, "how do I clean the filter on my {P}, please send the steps",
       "what temperature setting do you recommend for everyday use",
       "how do I change the language on the display"),
    sc("plain", ALL, "can you tell me the status of my earlier request",
       "Any update on my order? It was supposed to arrive last week",
       "I want to cancel my order and get a refund"),
    sc("indirect", ALL, "my {P} is not working properly",
       "something is wrong with it, please send someone",
       "I have a problem with my {P}"),
    sc("indirect", ALL, "need repair",
       "it's faulty, please fix",
       "{P} has an issue, help"),
    sc("indirect", ALL, "it stopped working yesterday",
       "not working since morning",
       "it isn't doing what it should"),
    sc("plain", ALL, "asdfghjkl",
       "test message please ignore",
       "hello is anyone there",
       "What's the weather like in Delhi today"),
    sc("multilingual", ALL, "bahut accha service, dhanyavaad",
       "gracias por la reparacion, todo bien ahora",
       "merci beaucoup pour votre aide"),
],
}

# --------------------------------------------------------------------------
# Wrappers: label-neutral context so real-world clutter is not a label signal.
# --------------------------------------------------------------------------
OPENERS = ["", "", "", "", "Hi, ", "Hello, ", "Hi team, ", "Dear support, ",
           "Good morning. ", "Please help: ", "Complaint: ", "Urgent - ",
           "To whom it may concern, ", "Hello customer care, ", "Re: my {P}. "]
CLOSERS = ["", "", "", "", " Please advise.", " Thanks.", " Thanks in advance.",
           " Waiting for your reply.", " Kindly send someone.", " Please help.",
           " Regards, Anil", " Thanks, Priya", " - Mark", " pls fix asap",
           " This is really frustrating.", " Sent from my iPhone"]
DETAILS = [
    "I bought it from the local dealer two years ago.",
    "We are a family of five so we use it daily.",
    "It is installed in the kitchen next to the window.",
    "My brother-in-law thinks it is the weather.",
    "I was travelling last week so only noticed now.",
    "The warranty card is somewhere in the drawer.",
    "We moved into this flat in March.",
    "I have already told the building manager.",
    "Our neighbour has the same model.",
    "It was a gift from my parents.",
    "I called twice before but nobody picked up.",
    "It is a white one, about four years old.",
    "This happened right after we came back from holiday.",
    "I'm not very technical so excuse my description.",
    "We had guests staying over at the time.",
    "The model number is somewhere on the back, I can check.",
]
FILLER = [
    "I know you must get many of these messages every day, but I have tried to sort it out myself first and I would really appreciate a proper reply this time.",
    "My husband works night shifts and I am at home with the kids most of the day, so a morning slot is easier for any visit.",
    "We renovated the kitchen last year and everything else in it has been working without any trouble at all until now.",
    "A friend suggested I contact you directly instead of the dealer, because the dealer has not been very helpful so far.",
    "I have attached nothing because I could not work out how to upload photos, but I can describe whatever you need.",
    "The last time I used your service the person on the phone was very nice, so I am hoping for the same experience again.",
    "I am writing this on my phone while standing in the kitchen, so please forgive the typing and any strange formatting.",
    "It is the festive season and we have relatives arriving at the weekend, which is why this is becoming urgent for us.",
]
_TEXTSPEAK = {"please": "pls", "thanks": "thx", "because": "coz", "you": "u",
              "your": "ur", "cannot": "cant", "can't": "cant", "won't": "wont",
              "doesn't": "doesnt", "isn't": "isnt", "don't": "dont",
              "before": "b4", "with": "wid"}
_TS_RE = re.compile(r"\b(?:" + "|".join(
    re.escape(k) for k in sorted(_TEXTSPEAK, key=len, reverse=True)) + r")\b", re.I)
_WORD_RE = re.compile(r"^([A-Za-z]+)([^A-Za-z]*)$")
_NEIGH = {}
for _row in ("qwertyuiop", "asdfghjkl", "zxcvbnm"):
    for _i, _c in enumerate(_row):
        _NEIGH[_c] = ([_row[_i - 1]] if _i > 0 else []) + (
            [_row[_i + 1]] if _i < len(_row) - 1 else [])


def _end(s):
    s = s.strip()
    return s if s and s[-1] in ".!?" else s + "."


def _cap(s):
    return s[:1].upper() + s[1:]


def _typo(word, rng):
    op = rng.choice(("swap", "drop", "dup", "sub"))
    i = rng.randrange(1, len(word) - 1)
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    near = _NEIGH.get(word[i].lower())
    return word if not near else word[:i] + rng.choice(near) + word[i + 1:]


def _typos(text, rng, k):
    toks = text.split(" ")
    idx = [i for i, t in enumerate(toks)
           if _WORD_RE.match(t) and len(_WORD_RE.match(t).group(1)) >= 4]
    for _ in range(k):
        if not idx:
            break
        j = rng.choice(idx)
        m = _WORD_RE.match(toks[j])
        toks[j] = _typo(m.group(1), rng) + m.group(2)
    return " ".join(toks)


def _noise(text, rng):
    r = rng.random()
    if r < 0.35:
        return text
    if r >= 0.75:  # heavy
        text = _typos(text, rng, rng.randint(1, 4))
        if rng.random() < 0.5:
            text = _TS_RE.sub(lambda m: _TEXTSPEAK[m.group(0).lower()]
                              if rng.random() < 0.7 else m.group(0), text)
        c = rng.random()
        text = text.lower() if c < 0.45 else text.upper() if c < 0.57 else text
        if rng.random() < 0.5:
            text = re.sub(r"[.,;!]", "", text)
    else:  # mild
        if rng.random() < 0.5:
            text = _typos(text, rng, rng.randint(0, 1))
        if rng.random() < 0.35:
            text = text.lower()
        if rng.random() < 0.35:
            text = text.rstrip(".!")
    return text.strip()


def _compose(core, noun, rng):
    opener = rng.choice(OPENERS).replace("{P}", noun)
    closer = rng.choice(CLOSERS)
    core = _cap(core) if rng.random() < 0.7 else core
    s = rng.random()
    if s < 0.42:
        return opener + _end(core) + closer
    if s < 0.62:
        return opener + _end(rng.choice(DETAILS)) + " " + _end(core) + closer
    if s < 0.84:
        return opener + _end(core) + " " + rng.choice(DETAILS) + closer
    sents = [rng.choice(DETAILS + FILLER) for _ in range(rng.randint(4, 8))]
    sents.insert(rng.randint(0, len(sents)), _end(_cap(core)))
    return (opener + " ".join(sents))[:2000]


def _assign_splits(scens, rng):
    """Split SCENARIOS (not rows). Each non-plain kind with >=2 scenarios is
    spread across splits: one to test, one to dev (if >=3), one kept in train."""
    n = len(scens)
    n_eval = max(2, round(0.2 * n))
    order = list(range(n))
    rng.shuffle(order)
    by_kind = defaultdict(list)
    for i in order:
        by_kind[scens[i][0]].append(i)
    test, dev, reserved = [], [], set()
    for kind, idxs in by_kind.items():
        if kind == "plain" or len(idxs) < 2:
            continue
        reserved.add(idxs[-1])          # guaranteed train
        test.append(idxs[0])
        if len(idxs) >= 3:
            dev.append(idxs[1])
    pool = [i for i in order if i not in test and i not in dev and i not in reserved]
    while len(test) < n_eval and pool:
        test.append(pool.pop())
    while len(dev) < n_eval and pool:
        dev.append(pool.pop())
    split = {i: "train" for i in range(n)}
    for i in test:
        split[i] = "test"
    for i in dev:
        split[i] = "dev"
    return split

def _validate_bank():
    for label in LABELS:
        if label not in SCENARIOS:
            raise ValueError(f"no scenarios for {label}")
        for kind, prods, paras in SCENARIOS[label]:
            if not set(prods) <= set(VALID[label]):
                raise ValueError(f"{label}: invalid products {prods}")
            if len(paras) < 2:
                raise ValueError(f"{label}: scenario needs >= 2 paraphrases")


def generate_dataset(seed=SEED):
    """Return a list of row dicts. Fully deterministic for a given seed."""
    _validate_bank()
    seen, rows = set(), []
    n_sc = 0   
    for label in LABELS:
        scens = SCENARIOS[label]
        splits = _assign_splits(scens, random.Random(f"{seed}|split|{label}"))
        per = max(6, ROWS_PER_LABEL // len(scens))
        for idx, (kind, prods, paras) in enumerate(scens):
            sid = f"sc{n_sc:03d}"               
            n_sc += 1  
            rng = random.Random(f"{seed}|{sid}")
            made, attempts = 0, 0
            while made < per and attempts < per * 40:
                attempts += 1
                letter = rng.choice(prods)
                noun = rng.choice(NOUNS[letter])
                core = rng.choice(paras).replace("{P}", noun).replace("{C}", rng.choice(CODES))
                text = _noise(_compose(core, noun, rng), rng)
                if len(text) < 5:
                    text += " please help"
                key = re.sub(r"[^a-z0-9]+", "", text.lower())
                if key in seen:
                    continue
                seen.add(key)
                made += 1
                rows.append({"product_category": _P[letter], "complaint_text": text,
                             "fault_category": label, "split": splits[idx],
                             "scenario_id": sid, "kind": kind})
    random.Random(f"{seed}|shuffle").shuffle(rows)
    for n, r in enumerate(rows):
        r["id"] = f"S{n:05d}"
    # sanity: scenario never spans splits; lengths within the brief's limits
    owner = {}
    for r in rows:
        assert owner.setdefault(r["scenario_id"], r["split"]) == r["split"]
        assert 5 <= len(r["complaint_text"]) <= 2000
        assert r["fault_category"] in LABELS and r["product_category"] in PRODUCTS
    return rows


def write_csv(rows, path=OUT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def load_csv(path=OUT_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    table = defaultdict(Counter)
    for r in rows:
        table[r["fault_category"]][r["split"]] += 1
    print(f"{'label':30s} {'train':>6s} {'dev':>6s} {'test':>6s} {'total':>6s}")
    for label in LABELS:
        c = table[label]
        print(f"{label:30s} {c['train']:6d} {c['dev']:6d} {c['test']:6d} {sum(c.values()):6d}")
    print(f"{'ALL':30s} {sum(1 for r in rows if r['split']=='train'):6d} "
          f"{sum(1 for r in rows if r['split']=='dev'):6d} "
          f"{sum(1 for r in rows if r['split']=='test'):6d} {len(rows):6d}")
    print("kinds:", dict(Counter(r["kind"] for r in rows)))
    print("\nExamples (first 2 train rows per class):")
    for label in LABELS:
        shown = [r for r in rows if r["fault_category"] == label and r["split"] == "train"][:2]
        for r in shown:
            print(f"  [{label}] ({r['product_category']}) {r['complaint_text'][:140]}")


if __name__ == "__main__":
    data = generate_dataset()
    write_csv(data)
    print(f"wrote {len(data)} rows to {OUT_PATH}\n")
    summarize(data)