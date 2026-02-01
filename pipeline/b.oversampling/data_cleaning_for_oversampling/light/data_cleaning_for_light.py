import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Light.csv"


# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*(alexa trigger).*phrase.*", "IF a voice assistant is activated with a specific phrase"),
    (r".*every.*single.*day.*specific time.*", "IF every day at a specified time"),
    (r".*(ok google).*phrase.*", "IF a voice assistant is activated with a specific phrase"),
    (r".*sunset.*15 minutes.*", "IF shortly before sunset at the user's location"),
    (r".*specific days.*week.*time.*", "IF on specific days of the week at a specified time"),
    (r".*press.*button.*", "IF a button is pressed"),
    (r".*enter.*area.*", "IF the user enters a specified area"),
    (r".*sunrise.*15 minutes.*", "IF shortly before sunrise at the user's location"),
    (r".*weather condition.*(rain|snow|cloudy|clear).*", "IF a specific weather condition is met"),
    (r".*once.*year.*date.*time.*", "IF once a year at a specified date and time"),
    (r".*exit.*area.*", "IF the user exits a specified area"),
    (r".*alarm.*goes off.*", "IF an alarm goes off"),
    (r".*once an hour.*(:00|:15|:30|:45).*", "IF every hour at quarter intervals"),
    (r".*arlo.*motion.*", "IF a security camera detects motion"),
    (r".*tagged.*new photo.*", "IF the user is tagged in a new photo"),
    (r".*ring.*doorbell.*", "IF a doorbell is rung"),
    (r".*doorbell.*motion.*", "IF motion is detected at a doorbell"),
    (r".*timer.*goes off.*", "IF a timer goes off"),
    (r".*new membership.*channel.*", "IF a new channel membership is created"),
    (r".*new super chat.*", "IF a new super chat message is received"),
    (r".*today.*weather report.*", "IF today's weather report is available"),
    (r".*apilio.*event.*", "IF a custom automation event is received"),
    (r".*flic.*click.*", "IF a smart button is pressed"),
    (r".*wemo.*long press.*", "IF a smart switch is long-pressed"),
    (r".*new video.*public.*", "IF a subscribed user publishes a new video"),
    (r".*nest protect.*dangerous smoke.*", "IF a smoke detector detects dangerous smoke"),
    (r".*new sms.*android.*", "IF a new SMS is received"),
    (r".*new.*status.*facebook.*", "IF a new social media status is posted"),
    (r".*trigger@applet.ifttt.com.*email.*", "IF an email is received by IFTTT"),
    (r".*home.*set to away.*", "IF a home is set to away mode"),
    (r".*(ok google).*post.*tweet.*", "IF a voice assistant posts a social message"),
    (r".*hashtag.*email.*ifttt.*", "IF an email with a hashtag is received"),
    (r".*miss.*phone call.*android.*", "IF a phone call is missed"),
    (r".*tap.*gesture.*ifttt.*", "IF a custom gesture is performed"),
    (r".*google calendar.*keyword.*before.*", "IF shortly before a calendar event with a keyword"),
    (r".*new notification.*android.*", "IF a new app notification is received"),
    (r".*nest protect.*warning smoke.*", "IF a smoke detector detects warning smoke"),
    (r".*add.*item.*to do list.*", "IF an item is added to a to-do list"),
    (r".*time for prayer.*", "IF it is time for prayer"),
    (r".*temperature.*drops below.*", "IF temperature drops below a threshold"),
    (r".*motion event.*detected.*", "IF motion is detected"),
    (r".*new item.*feed.*", "IF a new feed item is published"),
    (r".*amazon prime music.*song.*played.*", "IF a music track is played"),
    (r".*camera.*motion.*", "IF a camera detects motion"),
    (r".*carbon dioxide.*rises above.*", "IF CO2 level exceeds a threshold"),
    (r".*nest protect.*dangerous carbon monoxide.*", "IF dangerous carbon monoxide is detected"),
    (r".*@mentioned.*tweet.*", "IF the user is mentioned in a tweet"),
    (r".*connects.*wifi.*android.*", "IF a device connects to WiFi"),
    (r".*wemo.*turned on.*", "IF a smart switch is turned on"),
    (r".*wemo.*motion sensor.*", "IF a motion sensor detects motion"),
    (r".*google calendar.*before.*event.*", "IF shortly before any calendar event"),
    (r".*audio event.*detected.*", "IF an audio event is detected"),
    (r".*home.*set to home.*", "IF a home is set to home mode"),
    (r".*space station.*passes over.*", "IF the ISS passes over a location"),
    (r".*new follower.*channel.*", "IF a new channel follower is added"),
    (r".*spotify.*save.*track.*", "IF a music track is saved"),
    (r".*tomorrow.*weather.*forecast.*", "IF tomorrow's weather forecast changes"),
    (r".*skybell.*button.*", "IF a doorbell button is pressed"),
    (r".*twitter user.*tweets.*", "IF a specific user posts a tweet"),
    (r".*alexa.*to do list.*", "IF a voice assistant reads a to-do list"),
    (r".*fitbit.*sleep.*", "IF new sleep data is logged"),
    (r".*wemo.*turned off.*", "IF a smart switch is turned off"),
    (r".*withings.*motion.*", "IF a smart camera detects motion"),
    (r".*ihome.*button.*", "IF a smart button is pressed"),
    (r".*temperature.*rises above.*", "IF temperature rises above a threshold"),
    (r".*plage tarifaire.*commence.*", "IF an energy tariff period starts"),
    (r".*christmas.*", "IF it is Christmas"),
    (r".*new reminder.*", "IF a new reminder is added"),
    (r".*miss.*call.*number.*", "IF a call from a specific number is missed"),
    (r".*disconnects.*wifi.*android.*", "IF a device disconnects from WiFi"),
    (r".*daily step goal.*", "IF a daily step goal is achieved"),
    (r".*share.*photo.*instagram.*", "IF a photo is shared on social media"),
    (r".*android.*plugged in.*", "IF a device is plugged in"),
    (r".*battery.*drops below 15.*", "IF battery level is low"),
    (r".*smartthings.*motion.*", "IF a smart device detects motion"),
    (r".*wyze.*motion.*", "IF a security camera detects motion"),
    (r".*abode.*alarm.*active.*", "IF a security system is activated"),
    (r".*blink.*motion.*", "IF a security camera detects motion"),
    (r".*during alarms.*", "IF an alarm is active"),
    (r".*new photo.*camera roll.*", "IF a new photo is added to the gallery"),
    (r".*new track.*playlist.*", "IF a track is added to a playlist"),
    (r".*tweet.*matches search.*", "IF a tweet matches a search query"),
    (r".*new user.*following.*", "IF a new follower is added"),
    (r".*nest protect.*warning carbon monoxide.*", "IF warning carbon monoxide is detected"),
    (r".*place.*phone call.*android.*", "IF a phone call is made"),
    (r".*post.*tweet.*hashtag.*", "IF a tweet with a hashtag is posted"),
    (r".*netatmo.*rain.*", "IF rain is detected"),
    (r".*skybell.*motion.*", "IF a doorbell detects motion"),
    (r".*awair.*carbon dioxide.*above.*", "IF CO2 level exceeds a threshold"),
    (r".*new super sticker.*", "IF a new super sticker is received"),
    (r".*out for delivery.*", "IF an order is out for delivery"),
    (r".*device.*connects.*google wifi.*", "IF a device connects to WiFi"),
    (r".*feed.*contains.*keyword.*", "IF a feed item contains a keyword"),
    (r".*specified camera.*motion.*", "IF a camera detects motion"),
    (r".*stream.*going live.*", "IF a followed channel starts streaming"),
    (r".*device.*turned on.*", "IF a device is turned on"),
    (r".*smoke detector.*smoke alarm.*", "IF a smoke detector detects smoke"),
    (r".*electricity.*cheap.*denmark.*", "IF electricity prices are low"),
    (r".*electricity.*cheap.*norway.*", "IF electricity prices are low"),
    (r".*click.*flic.*", "IF a button is pressed"),
    (r".*edit.*to do list.*", "IF a to-do list item is edited"),
    (r".*cookit.*finished.*", "IF a smart cooking device finishes successfully"),
    (r".*cookit.*turned on.*", "IF a smart cooking device is turned on"),
    (r".*binary switch.*turned on.*", "IF a binary switch is turned on"),
    (r".*freezer.*door.*opened.*", "IF a freezer door is opened"),
    (r".*withings.*sleep.*out of bed.*", "IF a sleep tracking device detects the user leaving the bed"),
    (r".*(homeseer).*turned on.*", "IF a connected device is turned on"),
    (r".*(knx).*turned on.*", "IF a connected device is turned on"),
    (r".*security panel.*armed.*disarmed.*", "IF a security system is armed or disarmed"),
    (r".*fibaro.*button.*pressed.*", "IF a button is pressed"),
    (r".*co2.*intensity.*lowest.*", "IF carbon intensity is lowest within a time window"),
    (r".*electricity price.*lowest.*", "IF electricity prices are low"),
    (r".*exchanged power.*threshold.*", "IF power usage satisfies a threshold condition"),
    (r".*telegram.*key phrase.*|.*@ifttt.*telegram.*", "IF a message containing a key phrase is sent"),
    (r".*google calendar.*ending.*keyword.*", "IF shortly before or after a calendar event with a keyword"),
    (r".*sunrise.*15 minutes.*", "IF shortly before sunrise at the user's location"),
    (r".*lawnmower.*button.*pressed.*", "IF a button is pressed"),
    (r".*hours.*electricity.*denmark.*cheap.*", "IF electricity prices are low"),
    (r".*hours.*electricity.*norway.*cheap.*", "IF electricity prices are low"),
    (r".*fibaro.*home center.*button.*pressed.*", "IF a button is pressed"),
    (r".*google calendar.*ending time.*keyword.*", "IF shortly before or after a calendar event with a keyword"),
    (r".*within 15 minutes.*", "IF shortly before sunrise at the user's location"),
    (r".*smart button.*lawnmower.*pressed.*once.*|.*fibaro.*", "IF a button is pressed"),
# Lock / Door / Access
(r".*lock.*unlocked.*specific person.*|.*lock.*unlocked.*auto unlock.*|.*door.*unlocked.*",
 "IF a door or lock is unlocked"),

(r".*lock.*locked.*specific person.*|.*specified lock.*locked.*|.*lock is locked.*",
 "IF a door or lock is locked"),

(r".*door.*opened.*|.*door is open.*|.*garage.*open.*",
 "IF a door is opened"),

(r".*door.*closed.*|.*contact sensor.*closes.*",
 "IF a door is closed"),


# Voice / Speech
(r".*orion group.*says.*phrase.*|.*says.*particular word.*",
 "IF a voice assistant detects a spoken phrase"),


# Temperature / Thermostat
(r".*thermostat.*heating mode.*|.*set to.*heat mode.*",
 "IF a thermostat changes heating mode"),

(r".*temperature.*above.*value.*|.*exceeds.*temperature.*",
 "IF temperature rises above a threshold"),

(r".*temperature.*below.*threshold.*|.*falls below.*",
 "IF temperature drops below a threshold"),


# Appliance cycles
(r".*dryer cycle.*complete.*|.*dryer.*finished.*",
 "IF a dryer cycle finishes"),

(r".*washer cycle.*complete.*|.*dishwasher.*finishes.*cycle.*",
 "IF a washing cycle finishes"),

(r".*oven.*turns off.*timer.*|.*cooktop.*timer.*elapsed.*",
 "IF a cooking timer ends"),


# WiFi / Network
(r".*android.*connects.*wifi.*|.*device.*connects.*wifi.*|.*router.*connects.*",
 "IF a device connects to WiFi"),

(r".*android.*disconnects.*wifi.*|.*disconnects.*wifi.*",
 "IF a device disconnects from WiFi"),


# Calendar / Meetings
(r".*meeting.*starts.*|.*scheduled meeting.*starts.*",
 "IF a scheduled meeting starts"),

(r".*meeting.*ends.*|.*scheduled meeting.*ends.*",
 "IF a scheduled meeting ends"),

(r".*google calendar.*before.*event.*|.*minutes before.*calendar.*",
 "IF shortly before a calendar event"),


# Presence / Home arrival
(r".*family member.*come home.*|.*arrives home.*identified.*",
 "IF someone arrives home"),

(r".*family member.*left.*|.*everyone has left.*",
 "IF everyone leaves home"),

(r".*presence.*no longer detected.*",
 "IF user presence is no longer detected"),


# Security / Alarm
(r".*alarm panel.*armed.*|.*alarm is armed.*|.*mode is armed.*",
 "IF a security system is armed"),

(r".*alarm.*disarmed.*",
 "IF a security system is disarmed"),

(r".*receives an alarm.*|.*security panel.*alarm.*|.*eight alarm.*",
 "IF a security alarm is triggered"),


# Environmental sensors
(r".*wind speed.*above.*|.*wind.*rises above.*",
 "IF wind speed exceeds a threshold"),

(r".*humidity.*rises above.*",
 "IF humidity exceeds a threshold"),

(r".*air quality.*below.*|.*air quality.*unhealthy.*|.*filtrete.*air.*",
 "IF air quality is poor"),

(r".*pollen.*above.*",
 "IF pollen level is high"),

(r".*noise.*above.*decibel.*|.*sound is detected.*",
 "IF noise exceeds a threshold"),

(r".*uv index.*above.*",
 "IF UV index is high"),

(r".*light sensor.*lux.*",
 "IF ambient light level crosses a threshold"),


# CO2 / Emissions
(r".*co2.*drops below.*|.*co2 emissions.*low.*",
 "IF carbon emissions are low"),

(r".*co2.*rises above.*",
 "IF CO2 level exceeds a threshold"),


# Media / Social
(r".*new podcast episode.*|.*iot podcast.*",
 "IF a new podcast episode is published"),

(r".*like.*video.*youtube.*",
 "IF a video is liked"),

(r".*post.*new tweet.*|.*publish.*tweet.*",
 "IF a tweet is posted"),

(r".*upload.*new.*video.*youtube.*",
 "IF a new video is uploaded"),

(r".*wordpress.*publish.*post.*",
 "IF a blog post is published"),


# Email / IFTTT mail
(r".*send ifttt.*email.*trigger@applet.ifttt.com.*hashtag.*|.*send ifttt any email.*",
 "IF an email is received by IFTTT"),


# Smart buttons / Watches
(r".*festina watch.*pusher.*|.*kronaby watch.*pusher.*|.*lotus watch.*pusher.*",
 "IF a smartwatch button is pressed"),


# Devices / Power
(r".*ewelink.*switch.*on or off.*|.*device.*turned off.*|.*homeseer.*turned off.*",
 "IF a connected device changes power state"),

(r".*wemo.*switches on.*|.*wemo.*switches off.*|.*wemo.*standby.*",
 "IF a smart switch changes state"),


# Orders / Delivery
(r".*out for delivery.*|.*quality-checked.*order.*",
 "IF an order status changes"),


# Voicemail / Calls / SMS
(r".*new voicemail.*|.*leave any voicemail.*",
 "IF a voicemail is received"),

(r".*incoming call.*|.*receive.*call.*",
 "IF an incoming call is received"),

(r".*receive.*sms.*|.*send.*sms.*matches.*",
 "IF an SMS message is sent or received"),


# Health / Fitness
(r".*daily distance goal.*|.*do not achieve.*daily goal.*",
 "IF a fitness goal is evaluated"),

(r".*withings.*get into bed.*|.*sleep.*in bed.*",
 "IF sleep activity is detected"),


# Energy / Grid
(r".*price of electricity changes.*|.*peak rates.*start.*end.*",
 "IF electricity pricing changes"),

(r".*renewable generation.*changes.*",
 "IF renewable energy production changes"),


# Misc IoT
(r".*litter robot.*event.*|.*clean cycle.*completes.*",
 "IF a robot device completes a task"),

(r".*cloudbit.*input signal.*",
 "IF an IoT device receives an input signal"),

(r".*vibrations.*tag.*detects.*",
 "IF vibration is detected"),

(r".*spreadsheet.*cell.*updated.*",
 "IF a spreadsheet cell is updated"),

(r".*routine is activated.*",
 "IF an automation routine is activated"),

(r".*task is completed.*",
 "IF a task is completed"),

(r".*astronaut enters space.*",
 "IF a space event occurs"),

# -------------------------
# Locks / Doors / Access
# -------------------------
(r".*(auto unlock|unlock.*door|lock is unlocked|door is unlocked|locks the door).*",
 "IF a door or lock is unlocked"),

(r".*(door is locked|lock.*locked|trigger.*door is locked|unlocks the door).*",
 "IF a door or lock is locked"),

(r".*(contact sensor.*open|door opens|door is open|window is open).*",
 "IF a door or window is opened"),

(r".*(door left open|left open for).*",
 "IF a door is left open for a period"),


# -------------------------
# Motion / Presence / Camera
# -------------------------
(r".*(motion is detected|detects motion|movement detected|pir.*detects|sensor.*motion).*",
 "IF motion is detected"),

(r".*(motion.*no longer detected|becomes clear).*",
 "IF motion stops being detected"),

(r".*(detects a person|person detected|human detected|face id|recognizes face).*",
 "IF a person is detected"),

(r".*(pet is detected|vehicle is detected).*",
 "IF an object is detected"),

(r".*(doorbell.*motion|camera.*motion).*",
 "IF camera motion is detected"),


# -------------------------
# Thermostat / Temperature
# -------------------------
(r".*(temperature.*exceeds|above.*threshold|rises above|local room temperature).*",
 "IF temperature rises above a threshold"),

(r".*(temperature.*below|drops below).*",
 "IF temperature drops below a threshold"),

(r".*(set temperature.*changes|set temperature.*exceeds).*",
 "IF thermostat temperature setting changes"),

(r".*(thermostat.*away mode|switches to away|home mode).*",
 "IF thermostat changes mode"),

(r".*(thermostat.*error).*",
 "IF a thermostat error occurs"),


# -------------------------
# Appliance / Cycles
# -------------------------
(r".*(dryer.*cycle.*end|washer.*finishes|laundry cycle ends|wash cycle).*",
 "IF a washing or drying cycle finishes"),

(r".*(30 minutes after.*cycle).*",
 "IF some time has passed after a cycle ends"),

(r".*(oven.*timer.*elapsed|cooktop.*timer|oven.*finished).*",
 "IF a cooking timer ends"),

(r".*(coffee.*finished|kettle.*finished).*",
 "IF a beverage device finishes"),


# -------------------------
# Network / Connectivity
# -------------------------
(r".*(connects.*router|connects.*wifi|asus router|tp-link|d-link).*",
 "IF a device connects to a network"),

(r".*(disconnects.*router|disconnects.*wifi).*",
 "IF a device disconnects from a network"),

(r".*(user detected.*network|no one.*network).*",
 "IF network presence changes"),


# -------------------------
# Security / Alarm
# -------------------------
(r".*(alarm.*triggered|alarm sounds|alarm event occurs|detects smoke|co alarm).*",
 "IF a security alarm is triggered"),

(r".*(alarm.*dismissed|disarmed|mode is disarmed).*",
 "IF a security system is disarmed"),

(r".*(system.*armed|away mode.*armed).*",
 "IF a security system is armed"),

(r".*(intrusion detected|detects intrusion).*",
 "IF an intrusion is detected"),


# -------------------------
# Environmental / Air / Water
# -------------------------
(r".*(air quality.*(poor|moderate|good)|aqi updated|pollution|air sensors).*",
 "IF air quality changes"),

(r".*(humidity.*above|humidity.*below).*",
 "IF humidity crosses a threshold"),

(r".*(co2.*above|co2.*below|carbon dioxide).*",
 "IF CO2 level crosses a threshold"),

(r".*(leak detected|water leak).*",
 "IF a water leak is detected"),

(r".*(smoke detected|smoke sensor).*",
 "IF smoke is detected"),

(r".*(wind.*above|gust speed).*",
 "IF wind speed exceeds a threshold"),

(r".*(rain measurement|start of rain).*",
 "IF rain is detected"),


# -------------------------
# Media / Social / Content
# -------------------------
(r".*(new tweet|like a tweet|matches your search query).*",
 "IF a tweet activity occurs"),

(r".*(facebook.*post|post a new photo|link post).*",
 "IF a Facebook post is published"),

(r".*(new blog post|wordpress.*publish).*",
 "IF a blog post is published"),

(r".*(youtube|vimeo|video uploaded|live event).*",
 "IF a video event occurs"),

(r".*(spotify.*track|amazon prime music).*",
 "IF music is played or saved"),


# -------------------------
# Messages / Commands
# -------------------------
(r".*(/ifttt command|send ifttt|ifttt-command).*",
 "IF an IFTTT command is received"),

(r".*(new message|posts a new message).*",
 "IF a message is posted"),

(r".*(sms.*sent|send an sms|tagged sms).*",
 "IF an SMS is sent"),

(r".*(incoming call|answer a phone call).*",
 "IF a phone call occurs"),

(r".*(voicemail).*",
 "IF a voicemail is received"),


# -------------------------
# Finance / Orders / Business
# -------------------------
(r".*(order.*prepped|order.*ready|quality-checked|in the oven).*",
 "IF an order status changes"),

(r".*(spend money|transaction goes through).*",
 "IF a transaction occurs"),

(r".*(child.*money added|child.*money removed).*",
 "IF child account balance changes"),

(r".*(opportunity.*closed won|deal posted).*",
 "IF a business deal changes status"),


# -------------------------
# Health / Sleep / Fitness
# -------------------------
(r".*(log.*weight|withings.*weight).*",
 "IF a new weight is logged"),

(r".*(sleep.*logged|logs sleep|everyone has gone to sleep).*",
 "IF sleep activity is detected"),

(r".*(daily goal|point goal|goal.*derailing).*",
 "IF a fitness goal status changes"),


# -------------------------
# Presence / Location / Geo
# -------------------------
(r".*(arrives home|arrives.*identified|specific person arrives).*",
 "IF someone arrives home"),

(r".*(vehicle.*enters|vehicle.*exits|geo-circle|virtual fence).*",
 "IF a geofence boundary is crossed"),

(r".*(enter a room|leave a room).*",
 "IF someone enters or leaves a room"),

(r".*(scheduled.*away|scheduled.*home|scheduled.*sleep).*",
 "IF scheduled presence conditions are met"),


# -------------------------
# Automation / Devices
# -------------------------
(r".*(routine is activated|automation runs|nexia automation).*",
 "IF an automation runs"),

(r".*(scene is changed|scene is started|knx scene).*",
 "IF a scene is activated"),

(r".*(device.*turned on|device.*turned off|switch.*on|switch.*off).*",
 "IF a device changes power state"),

(r".*(eco mode|cool mode|energy saving mode).*",
 "IF a device changes energy mode"),

(r".*(hub.*state|system mode changes).*",
 "IF system mode changes"),


# -------------------------
# Sensors / Power / Battery
# -------------------------
(r".*(low battery|nearly out of battery|battery levels fall).*",
 "IF battery level is low"),

(r".*(power.*rises above|energy use threshold|cost rises).*",
 "IF power consumption exceeds a threshold"),

(r".*(ambient air pressure).*",
 "IF air pressure exceeds a threshold"),


# -------------------------
# Buttons / Wearables
# -------------------------
(r".*(button is pressed|press the stone|fetch-it|thinga).*",
 "IF a smart button is pressed"),

(r".*(festina|kronaby|lotus|jaguar).*pusher.*",
 "IF a smartwatch button is pressed"),


# -------------------------
# Data / Cloud / Platforms
# -------------------------
(r".*(spreadsheet.*cell|new record|sobject).*",
 "IF a data record is updated"),

(r".*(github.*notification|issue is closed).*",
 "IF a GitHub activity occurs"),

(r".*(survey response).*",
 "IF a survey response is received"),

(r".*(salesforce).*",
 "IF a Salesforce event occurs"),


# -------------------------
# News / External Events
# -------------------------
(r".*(breaking news|disease outbreak|president signs).*",
 "IF a major news event occurs"),

(r".*(weather.*forecast).*",
 "IF weather forecast changes"),

(r".*(bart.*delayed).*",
 "IF public transport is delayed"),

# =========================
# Platform / Webhooks / Events
# =========================
(r".*(apilio|particle\.publish|ifttt event|interesting event|mesh app).*receives.*event|sends an event|matches the specified criteria.*",
 "IF an external platform event is received"),

(r".*(event matches|given combination.*device.*zone.*action).*",
 "IF a system event matches specific criteria"),


# =========================
# Smartwatches / Buttons
# =========================
(r".*(festina|kronaby|lotus|jaguar).*pusher|watch.*pusher.*",
 "IF a smartwatch button is pressed"),


# =========================
# Recommendations / Reminders / Tasks
# =========================
(r".*(recommend.*applet|new applet).*daily.*",
 "IF a daily recommendation is generated"),

(r".*(reminder.*completed|task.*completed|to do list|task in a given list).*",
 "IF a task or reminder is completed"),

(r".*(submit.*note|new note).*",
 "IF a new note is created"),


# =========================
# Voice Assistants / Queries
# =========================
(r".*(alexa.*shopping list|ask alexa.*shopping|team.*next game).*",
 "IF a voice assistant query is made"),


# =========================
# Music / Media
# =========================
(r".*(spotify.*save.*track|save.*music).*",
 "IF music is saved"),

(r".*(take a photo.*front camera|rear camera|take a photo).*",
 "IF a photo is taken"),


# =========================
# Cooking / Appliances / Timers
# =========================
(r".*(cooktop.*timer|oven.*timer|timer.*elapsed).*",
 "IF a cooking timer ends"),

(r".*(preheating phase.*finished|fast preheat).*",
 "IF appliance preheating finishes"),

(r".*(oven.*turned on|hood.*turned on|hood.*turned off).*",
 "IF a kitchen appliance changes state"),


# =========================
# Thermostat / Home Modes
# =========================
(r".*(smart home|smart away|comfort profiles).*",
 "IF thermostat comfort mode changes"),

(r".*(away mode.*(on|off|canceled)|enters away|exits away|home enters away).*",
 "IF home presence mode changes"),

(r".*(abode.*mode|system mode is changed).*",
 "IF security system mode changes"),


# =========================
# Doors / Sensors / Openings
# =========================
(r".*(refrigerator|freezer).*door.*open|device.*opened|smartthings.*opened.*",
 "IF a door or device is opened"),

(r".*(gate.*opening|gate start opening).*",
 "IF a gate starts opening"),


# =========================
# Security / Alarms / Intrusion
# =========================
(r".*(alarm.*detected|alarm panel|alarm state|smart alert|fire alarm).*",
 "IF an alarm is detected"),

(r".*(intrusion|adaptors detect intrusion|detects intrusion).*",
 "IF an intrusion is detected"),

(r".*(otiom tag.*alarm).*",
 "IF a tracking tag enters alarm state"),


# =========================
# Motion / Person / Face Detection
# =========================
(r".*(human detected|person detected|face detected|unknown person|specific person).*",
 "IF a person is detected"),

(r".*(motion sensor|detects movement|detects motion|nearby movement).*",
 "IF motion is detected"),

(r".*(vehicle detected|detects a vehicle).*",
 "IF a vehicle is detected"),


# =========================
# Environment / Air / CO2 / Carbon
# =========================
(r".*(co₂|co2).*exceed|co₂.*below|carbon dioxide.*threshold.*",
 "IF CO2 level crosses a threshold"),

(r".*(air quality alert|aqi updated|aqi of your device).*",
 "IF air quality changes"),

(r".*(carbon intensity|carbon.*clean|dirty).*",
 "IF carbon impact indicators change"),


# =========================
# Power / Energy / Pricing
# =========================
(r".*(electricity price|dynamic pricing|average price changes).*",
 "IF electricity price changes"),

(r".*(peak time savings|ohmhour).*",
 "IF an energy-saving event starts"),

(r".*(price.*drops|price.*rises).*stock.*",
 "IF stock price crosses a threshold"),


# =========================
# Robots / Automation Jobs
# =========================
(r".*(robot.*starts|robot.*finishes|robot.*completes|starts cleaning).*",
 "IF a robot starts or completes a task"),

(r".*(tecan instrument.*state changes|run.*completed).*",
 "IF an industrial device changes state"),


# =========================
# Location / Social Check-ins
# =========================
(r".*(foursquare|swarm).*check in.*",
 "IF a location check-in occurs"),

(r".*(post on facebook.*location).*",
 "IF a social post is published at a location"),


# =========================
# Blogs / News / Deals
# =========================
(r".*(wordpress.*publish|new blog post).*",
 "IF a blog post is published"),

(r".*(ifttt blog|tagged.*updates).*",
 "IF a platform update is published"),

(r".*(slickdeals|new deal).*",
 "IF a deal is published"),

(r".*(best sellers list|new book).*",
 "IF a bestseller list is updated"),


# =========================
# Devices / Power State
# =========================
(r".*(product turns on|product turns off|device.*turns on.*off).*",
 "IF a device changes power state"),

(r".*(device in your house.*on.*off).*",
 "IF any home device changes state"),


# =========================
# Health / Sleep / Weight
# =========================
(r".*(new weight|withings.*weight).*",
 "IF a new weight is logged"),

(r".*(log sleep|sleep above target).*",
 "IF sleep activity exceeds target"),

(r".*(screen unlocks.*exceeds).*",
 "IF smartphone usage exceeds a limit"),


# =========================
# Water / Irrigation / Leaks
# =========================
(r".*(watering time starts|watering event).*",
 "IF irrigation starts"),

(r".*(leak detected).*",
 "IF a water leak is detected"),


# =========================
# Presence / Network / Detection
# =========================
(r".*(user detected.*network|presence is detected).*",
 "IF presence is detected"),

(r".*(blue by adt.*activity).*",
 "IF home activity is detected"),


# =========================
# Misc / Generic Safety Net
# =========================
(r".*(incident gets assigned|new incident).*",
 "IF a system incident is assigned"),

(r".*(rust.*killed|game event).*",
 "IF a gaming event occurs"),

(r".*(caavo.*search query).*",
 "IF a media search is performed"),

(r".*(mui board.*night mode).*",
 "IF device night mode changes"),

(r".*(watering|irrigation).*",
 "IF watering activity occurs"),

# --- Orologi (Basta il brand) ---
(r".*watch.*", "IF the watch button is pressed"),

    # --- App e Servizi Specifici (Parola chiave unica) ---
    (r".*apilio.*", "IF IFTTT receives event from Apilio"),
    (r".*github.*", "IF new notification on GitHub"),
    (r".*wordpress.*", "IF new post on WordPress"),
    (r".*phyn.*", "IF Phyn detects an alert"),
    (r".*beseye.*", "IF Beseye camera detects human"),
    (r".*porkfolio.*", "IF money added to Porkfolio"),
    (r".*mesh app.*", "IF MESH tag signal received"),
    (r".*eight app.*", "IF Eight App mode toggled"),
    (r".*flics.*", "IF Flic button used"),
    (r".*rust.*", "IF killed in Rust game"),
    (r".*particle.*", "IF Particle event received"),
    (r".*livy.*", "IF Livy alarm activated"),
    (r".*spotcam.*", "IF SpotCam battery cover removed"),
    (r".*roxxter.*", "IF Roxxter robot started"),
    (r".*ge dryer.*", "IF GE Dryer finishes cycle"),
    (r".*smart bed.*", "IF smart bed alarm activated"),

    # --- Sicurezza e Sensori (Brand o Azione breve) ---
    (r".*swann.*", "IF Swann security device detects activity"),
    (r".*wyze.*sound.*", "IF Wyze Cam detects sound"),
    (r".*wyze.*smoke.*", "IF Wyze Cam detects smoke alarm"),
    (r".*nest.*batteries.*", "IF Nest Protect battery is low"),
    (r".*detects vibrations.*", "IF tag detects vibrations"),
    (r".*leak is detected.*", "IF water leak is detected"),
    (r".*smoke is detected.*", "IF smoke is detected"),
    (r".*co.*lower.*", "IF CO2 level drops below threshold"),
    (r".*person is detected.*", "IF a person is detected"),
    (r".*child lock.*", "IF child lock deactivated"),

    # --- Finanza (Stocks) ---
    (r".*stock.*drops.*", "IF stock price drops"),
    (r".*stock.*rises.*", "IF stock price rises"),

    # --- Social e Location (Check-in) ---
    (r".*friend checks in.*", "IF a friend checks in"),  # Prima questo per priorità
    (r".*check.*in.*", "IF you check in on Foursquare/Swarm"),

    # --- Vari / Generici ---
    (r".*recommends.*applet.*", "IF IFTTT recommends new Applet"),
    (r".*group decision.*", "IF group decision made"),
    (r".*time for fast.*", "IF it is time to fast"),
    (r".*android.*unplugged.*", "IF Android device unplugged"),
    (r".*spreadsheet.*", "IF spreadsheet updated"),
    (r".*louisville.*", "IF emergency notification in Louisville"),
    (r".*forecasted weather.*", "IF weather forecast changes"),
    (r".*scene.*changed.*", "IF room scene changed"),
    (r".*site.*modes.*", "IF site mode changes"),
    (r".*fourth of july.*", "IF it is Fourth of July"),
    (r".*nearby.*network.*", "IF user detected on network"),
    (r".*imminent derailment.*", "IF goal at risk (derailment)"),
    (r".*shopping list.*", "IF item added to shopping list"),
    (r".*parameter exceed.*", "IF parameter exceeds threshold"),

]

ACTION_RULES = [
(r".*change.*brightness.*wemo.*group.*|.*change.*brightness.*wemo.*light.*", ", THEN sets the brightness level of smart lights."),
(r".*sleep fader.*wemo.*group.*|.*sleep fader.*wemo.*light.*", ", THEN starts a sleep fade on smart lights."),
(r".*turns off.*light switch.*|.*turn off.*light switch.*", ", THEN turns off a light switch."),
(r".*turns on.*light switch.*|.*turn on.*light switch.*", ", THEN turns on a light switch."),
(r".*set.*scene.*hue.*|.*activate.*scene.*|.*nanoleaf.*scene.*|.*wiz.*scene.*|.*lutron.*scene.*", ", THEN activates a lighting scene."),
(r".*briefly.*hue.*off.*on.*", ", THEN briefly toggles the lights off and on."),
(r".*change.*brightness.*bulb.*|.*set.*brightness.*light.*|.*dim.*brighten.*", ", THEN adjusts the brightness of lights."),
(r".*change.*color temperature.*|.*specified color temperature.*", ", THEN sets the color temperature of lights."),
(r".*change.*color.*nanoleaf.*|.*change.*color.*hue.*|.*change.*color.*lights.*|.*set.*color.*", ", THEN changes the color of lights."),
(r".*random.*color.*|.*match.*dominant colors.*", ", THEN sets lights to a random or image-based color."),
(r".*dynamic mode.*|.*flashing.*|.*jumping.*|.*strobe.*", ", THEN enables a dynamic lighting mode."),
(r".*pulse.*gently.*|.*blink.*|.*breathe.*", ", THEN triggers a light animation effect."),
(r".*pre-defined scene.*", ", THEN sets lights to a predefined scene."),
(r".*switch off.*light.*", ", THEN switches off the lights."),
(r".*switch on.*light.*", ", THEN switches on the lights."),
(r".*temporarily.*nanoleaf.*solid.*blinking.*", ", THEN temporarily changes light color and restores it."),
(r".*toggle.*wemo.*|.*toggle.*hive.*|.*toggle.*hue.*|.*toggle.*lights.*", ", THEN toggles the lights on or off."),
(r".*turn.*wemo.*off.*on.*|.*turn.*wemo.*on.*off.*", ", THEN power-cycles a smart switch."),
(r".*turn off.*functional light.*hood.*|.*turn off.*ambient light.*hood.*", ", THEN turns off the hood lighting."),
(r".*turn on.*ambient light.*hood.*|.*turn on.*functional light.*hood.*", ", THEN turns on the hood lighting."),
(r".*turn off.*nanoleaf.*|.*turn off.*hue.*|.*turn off.*sengled.*|.*turn off.*selected light.*", ", THEN turns off smart lights."),
(r".*turn on.*nanoleaf.*|.*turn on.*hue.*|.*turn on.*sengled.*|.*turn on.*selected light.*", ", THEN turns on smart lights."),
(r".*color loop.*hue.*", ", THEN starts a color loop lighting effect."),
(r".*on/off.*led.*projector.*", ", THEN toggles the projector LED light."),
(r".*night light.*off.*", ", THEN turns off the night light."),
(r".*night light.*on.*", ", THEN turns on the night light."),
(r".*hive.*off.*duration.*", ", THEN turns off the light for a specified duration."),
(r".*hive.*on.*duration.*", ", THEN turns on the light for a specified duration."),
(r".*turn your lights off if.*on.*|.*turn your lights on if.*off.*", ", THEN toggles lights based on current state."),
(r".*turn your lights off.*|.*turns off.*wemo.*group.*|.*turns off.*wemo.*light.*", ", THEN turns off smart lights."),
(r".*turn your lights on.*|.*turns on.*wemo.*group.*|.*turns on.*wemo.*light.*", ", THEN turns on smart lights."),
(r".*adjust.*multiple.*lights.*shades.*activating.*scene.*lutron.*|.*lutron.*app.*scene.*",", THEN activates a lighting scene."),
(r".*turn.*wemo.*light switch.*on.*remain on.*|.*turn.*wemo.*switch.*on.*",", THEN turns on a light switch."),
(r".*turn.*wemo.*light switch.*off.*remain off.*|.*turn.*wemo.*switch.*off.*",", THEN turns off a light switch."),
(r".*set.*(your )?lights?.*specified brightness.*|.*set.*light.*specified brightness.*",", THEN adjusts the brightness of lights."),
(r".*change.*light.*brightness.*specified level.*|.*change.*lights?.*brightness.*level.*", ", THEN adjusts the brightness of lights."),
(r".*set.*dim level.*light.*|.*set.*dim.*level.*", ", THEN adjusts the brightness of lights."),
(r".*change.*color.*light bulb.*|.*change.*color.*bulb.*", ", THEN changes the color of lights."),
(r".*change.*lights?.*specified color.*|.*set.*lights?.*specified color.*", ", THEN changes the color of lights."),
(r".*apply.*scene.*wiz.*lights.*|.*wiz.*scene.*settings.*", ", THEN activates a lighting scene."),
]

# ===============================
# NORMALIZATION FUNCTIONS
# ===============================

def normalize_text(text, rules, unmatched_tag):

    if pd.isna(text):
        return text

    text = str(text).lower().strip()

    for pattern, normalized in rules:
        if re.search(pattern, text, re.IGNORECASE):
            return normalized

    return f"{unmatched_tag} {text}"


def normalize_trigger(text):
    return normalize_text(text, TRIGGER_RULES, "__UNMATCHED_TRIGGER__")


def normalize_action(text):
    return normalize_text(text, ACTION_RULES, "__UNMATCHED_ACTION__")


# ===============================
# PROCESS
# ===============================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

filtered_df = df

filtered_df = filtered_df[
    ~filtered_df["triggerDesc"].str.contains(
        r".*space.*|.*astronaut.*",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r"NaN",
        case=False,
        na=False
    )
]

# Salva il nuovo CSV
filtered_df.to_csv(INPUT_FILE, index=False)

print(f"File salvato: {INPUT_FILE}")
print(f"Righe originali: {len(df)}")
print(f"Righe dopo filtro: {len(filtered_df)}")


# Check columns
if "triggerDesc" not in df.columns:
    raise ValueError("Missing column: triggerDesc")

if "actionDesc" not in df.columns:
    raise ValueError("Missing column: actionDesc")


# Backup original
df["originalTrigger"] = df["triggerDesc"]
df["originalAction"] = df["actionDesc"]


# Normalize
print("Normalizing triggers...")
df["triggerDesc"] = df["triggerDesc"].apply(normalize_trigger)

print("Normalizing actions...")
df["actionDesc"] = df["actionDesc"].apply(normalize_action)


# ===============================
# SAVE OUTPUT
# ===============================

folder = os.path.dirname(INPUT_FILE)
base = os.path.basename(INPUT_FILE).replace(".csv", "")

output_file = os.path.join(
    folder,
    f"{base}_normalized_final.csv"
)

df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")

# ===============================
# VALIDATION / REPORT
# ===============================

print("\n========== NORMALIZATION REPORT ==========")

# Trigger stats
unique_triggers = df["originalTrigger"].dropna().unique()
unique_unmatched_triggers = (
    df[df["triggerDesc"].str.contains("__UNMATCHED_TRIGGER__", na=False)]
    ["originalTrigger"]
    .dropna()
    .unique()
)

total_unique_triggers = len(unique_triggers)
unmatched_unique_triggers = len(unique_unmatched_triggers)

unique_trigger_coverage = 100 * (total_unique_triggers - unmatched_unique_triggers) / total_unique_triggers

# Action stats
unique_actions = df["originalAction"].dropna().unique()
unique_unmatched_actions = (
    df[df["actionDesc"].str.contains("__UNMATCHED_ACTION__", na=False)]
    ["originalAction"]
    .dropna()
    .unique()
)

total_unique_actions = len(unique_actions)
unmatched_unique_actions = len(unique_unmatched_actions)

unique_action_coverage = 100 * (total_unique_actions - unmatched_unique_actions) / total_unique_actions

print("\n=== UNIQUE COVERAGE ===")

print(f"Unique triggers normalized: "
      f"{total_unique_triggers - unmatched_unique_triggers}/{total_unique_triggers} "
      f"({unique_trigger_coverage:.2f}%)")

print(f"Unique actions normalized: "
      f"{total_unique_actions - unmatched_unique_actions}/{total_unique_actions} "
      f"({unique_action_coverage:.2f}%)")


# === PRINT ALL UNMATCHED ===

print("\n========== ALL UNMATCHED TRIGGERS ==========\n")

unmatched_triggers_df = (
    df[df["triggerDesc"].str.contains("__UNMATCHED_TRIGGER__", na=False)]
    [["originalTrigger", "triggerDesc"]]
    .drop_duplicates()
)

if unmatched_triggers_df.empty:
    print("No unmatched triggers ✅")
else:
    for _, row in unmatched_triggers_df.iterrows():
        print(f"- ORIGINAL : {row['originalTrigger']}")
        print(f"  CURRENT  : {row['triggerDesc']}\n")


print("\n========== ALL UNMATCHED ACTIONS ==========\n")

unmatched_actions_df = (
    df[df["actionDesc"].str.contains("__UNMATCHED_ACTION__", na=False)]
    [["originalAction", "actionDesc"]]
    .drop_duplicates()
)

if unmatched_actions_df.empty:
    print("No unmatched actions ✅")
else:
    for _, row in unmatched_actions_df.iterrows():
        print(f"- ORIGINAL : {row['originalAction']}")
        print(f"  CURRENT  : {row['actionDesc']}\n")


# Remove useless columns
COLUMNS_TO_DROP = [
    "id",
    "triggerChannelTitle",
    "triggerTitle",
    "actionChannelTitle",
    "actionTitle",
    "deviceCategory",
    "originalTrigger",
    "originalAction"
]

df = df.drop(
    columns=COLUMNS_TO_DROP,
    errors="ignore"
)

df["freeOrLocked"] = "LOCKED"

df.to_csv(output_file, index=False)

print(f"Saved: {output_file}")

print("==========================================")
