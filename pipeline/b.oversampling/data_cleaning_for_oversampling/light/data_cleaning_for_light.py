import pandas as pd
import os
import re


# ===============================
# CONFIG
# ===============================

INPUT_FILE = "../../../../dataset/filtered_data/device_category_data/Light_normalized_for_oversampling.csv"

# ===============================
# NORMALIZATION RULES
# ===============================

TRIGGER_RULES = [
    (r".*facebook profile.*changes.*", "IF your Facebook profile information changes"),
    (r".*new record.*ifttt event.*sobject.*", "IF a new record is created in your IFTTT Event Sobject"),
    (r".*child.*money added.*", "IF money is added to a child account"),
    (r".*child.*money removed.*", "IF money is removed from a child account"),
    (r".*incident.*assigned.*current user.*", "IF a new incident is assigned to you"),
    (r".*tagged sms.*hashtag.*", "IF you send an SMS with a hashtag"),
    (r".*otiom tag.*alarm state.*", "IF the Otiom tag enters an alarm state"),
    (r".*motion sensor.*detects occupancy.*", "IF a motion sensor detects occupancy"),
    (r".*breaking news alert.*", "IF there is a breaking news alert"),
    (r".*peak time savings.*announced.*", "IF peak time savings hours are announced"),
    (r".*dryer cycle.*ends.*", "IF a dryer cycle ends"),
    (r".*wash cycle.*ends.*", "IF a wash cycle ends"),
    (r".*nest protect.*low batter.*", "IF Nest Protect detects low batteries"),
    (r".*stock.*closing price drops below.*", "IF a stock's closing price drops below a specified amount"),
    (r".*client device connects.*d-link router.*", "IF a client device connects to your D-Link router"),
    (r".*device.*turns on or off.*", "IF a device in your house turns on or off"),
    (r".*group decision.*direct action.*taken.*", "IF a group decision or direct action is taken"),
    (r".*new book.*best sellers list.*", "IF a new book is added to a best sellers list"),
    (r".*new photo.*android.*area.*", "IF a new photo is taken on your Android device in a specified area"),
    (r".*new post.*hottest.*subreddit.*", "IF a new post becomes one of the hottest in a subreddit"),
    (r".*reminder.*completed.*", "IF a reminder is completed in a specified list"),
    (r".*device connects.*tp-link router.*", "IF a previously-connected device connects to your TP-Link router"),
    (r".*device disconnects.*tp-link router.*", "IF a previously-connected device disconnects from your TP-Link router"),
    (r".*sighthound.*rule.*sends event.*", "IF a Sighthound Video rule sends an event"),
    (r".*sighthound video rule.*event.*", "IF a Sighthound Video rule sends an event"),
    (r".*smoke sensor.*senses smoke.*", "IF a smoke sensor detects smoke"),
    (r".*channel publishes.*video.*", "IF a specific channel publishes a video"),
    (r".*device connects.*asus router.*", "IF a specific device connects to your ASUS router"),
    (r".*tag detects vibrations.*", "IF a tag detects vibrations"),
    (r".*watering time starts.*", "IF a watering time starts for a device"),
    (r".*opportunity.*closed won.*", "IF an opportunity changes to 'Closed Won' stage"),
    (r".*friend checks in.*foursquare.*area.*", "IF any friend checks in on Foursquare in a specified area"),
    (r".*issue.*closed.*repository.*", "IF any issue is closed in a repository you own or collaborate on"),
    (r".*motion.*no longer detected.*", "IF motion is no longer detected in a room"),
    (r".*nasa.*astronomy picture.*day.*", "IF NASA posts a new astronomy picture of the day"),
    (r".*nasa.*image.*day gallery.*", "IF NASA posts to the image of the day gallery"),
    (r".*netatmo welcome.*detects.*person.*", "IF Netatmo Welcome detects a specific known person"),
    (r".*lock.*n.*go.*button.*twice.*", "IF someone locks the door with Lock 'n' Go"),
    (r".*unknown.*rings doorbell.*opener.*", "IF someone unknown rings the doorbell"),
    (r".*ambient air pressure rises.*", "IF the ambient air pressure rises above a specified value"),
    (r".*camera detects motion.*", "IF the camera detects motion"),
    (r".*door.*locked.*", "IF the door is locked"),
    (r".*humidity drops below.*", "IF the humidity drops below a specified percentage"),
    (r".*power rises above.*watts.*", "IF the power rises above a specified value in watts"),
    (r".*president signs.*bill.*law.*", "IF the president signs a new bill into law"),
    (r".*lock.*locked.*", "IF a specified lock is locked"),
    (r".*temperature.*weatherflow tempest.*drops below.*", "IF the temperature at your WeatherFlow Tempest station drops below a value"),
    (r".*temperature drops below.*", "IF the temperature drops below a chosen value"),
    (r".*temperature.*spotter rises above.*", "IF the temperature measured by your Spotter rises above a value"),
    (r".*temperature rises above.*", "IF the temperature rises above a chosen value"),
    (r".*wind speed rises above.*location.*", "IF the wind speed rises above a value in your location"),
    (r".*wind speed rises above.*", "IF the wind speed rises above a specified value"),
    (r".*new survey response.*", "IF there is a new survey response"),
    (r".*tomorrow.*forecasted low temperature drops.*", "IF tomorrow's forecasted low temperature drops below a value"),
    (r".*low battery.*spotcam.*", "IF a low battery event is detected by SpotCam"),
    (r".*person.*detected.*", "IF a person is detected"),
    (r".*virtual fence.*crossed.*", "IF a virtual fence is crossed"),
    (r".*spotcam ring battery cover.*removed.*", "IF SpotCam Ring battery cover is removed"),
    (r".*spotcam ring.*doorbell.*pressed.*", "IF SpotCam Ring video doorbell is pressed"),
    (r".*achieve.*daily point goal.*", "IF you achieve your daily point goal"),
    (r".*add.*item.*shopping list.*", "IF you add an item to your shopping list"),
    (r".*add money.*porkfolio.*", "IF you add money to your Porkfolio"),
    (r".*answer.*phone call.*android.*", "IF you answer a phone call on your Android device from a specified number"),
    (r".*ask alexa.*team.*next game.*", "IF you ask Alexa for a team's next game"),
    (r".*check in.*foursquare.*hashtag.*", "IF you check in on Foursquare with a hashtag"),
    (r".*create.*link post.*facebook.*", "IF you create a new link post on Facebook"),
    (r".*create.*status message.*facebook.*hashtag.*", "IF you create a new status message on Facebook with a hashtag"),
    (r".*create.*status message.*facebook page.*", "IF you create a new status message on your Facebook page"),
    (r".*long press.*logi button.*", "IF you do a long press on the Logi button"),
    (r".*short press.*logi button.*", "IF you do a short press on the Logi button"),
    (r".*do not achieve.*daily goal.*", "IF you do not achieve a daily goal"),
    (r".*favorite.*public photo.*flickr.*", "IF you favorite a public photo on Flickr"),
    (r".*follow.*new show.*spotify.*", "IF you follow a new show on Spotify"),
    (r".*new weight.*withings scale.*", "IF you have a new weight from your Withings scale"),
    (r".*post.*facebook.*location.*", "IF you post on Facebook at a specified location"),
    (r".*publish.*post.*wordpress.*", "IF you publish a new post on your WordPress blog"),
    (r".*send.*sms.*android.*", "IF you send an SMS on your Android device to a specified number"),
    (r".*share.*photo.*instagram.*hashtag.*", "IF you share a photo on Instagram with a hashtag"),
    (r".*share.*video.*instagram.*", "IF you share any new video on Instagram"),
    (r".*single press.*misfit flash button.*", "IF you single press your Misfit Flash button"),
    (r".*start.*live event.*vimeo.*", "IF you start a live event on Vimeo"),
    (r".*take.*photo.*rear camera.*", "IF you take a photo with the rear camera"),
    (r".*take.*photo.*do camera.*", "IF you take any new photo via Do Camera"),
    (r".*triple press.*misfit flash button.*", "IF you triple press your Misfit Flash button"),
    (r".*air purifier.*air quality.*", "IF your air purifier registers a specified air quality"),
    (r".*android.*disconnects.*wifi.*", "IF your Android device disconnects from any WiFi network"),
    (r".*android.*unplugged.*", "IF your Android device is unplugged"),
    (r".*d-link camera.*sound event.*", "IF your D-Link camera detects a sound event"),
    (r".*d-link water sensor.*leak.*", "IF your D-Link water sensor detects a leak event"),
    (r".*device.*nearly out.*battery.*", "IF your device is nearly out of battery"),
    (r".*dishwasher door.*opened.*", "IF your dishwasher door has been opened"),
    (r".*ecobee light switch.*turned off.*", "IF your Ecobee light switch is turned off"),
    (r".*ecobee light switch.*turned on.*", "IF your Ecobee light switch is turned on"),
    (r".*ge washer.*finishes.*cycle.*", "IF your GE washer finishes a cycle"),
    (r".*home exits.*away mode.*", "IF your home exits away mode"),
    (r".*hood.*turned off.*", "IF your hood is turned off"),
    (r".*hood.*turned on.*", "IF your hood is turned on"),
    (r".*misfit shine.*sleep.*duration above.*", "IF your Misfit Shine logs sleep above target hours"),
    (r".*myfox security system.*armed.*", "IF your MyFox security system is armed"),
    (r".*oven door.*closed.*", "IF your oven door has been closed"),
]

TRIGGER_RULES += [
    (r".*device.*set to eco mode.*", "IF the specified device is set to eco mode"),
    (r".*device.*set to heat mode.*", "IF the specified device is set to heat mode"),
    (r".*specified device.*turned off.*", "IF the specified device is turned off"),
    (r".*specified product turns off.*", "IF the specified product turns off"),
    (r".*specified product turns on.*", "IF the specified product turns on"),
    (r".*door sensor.*open or closed.*", "IF the state of a selected door sensor is open or closed"),
    (r".*system detects.*water leak.*", "IF the system detects a water leak"),
    (r".*new user voicemail.*", "IF there is a new user voicemail"),
    (r".*receive.*phone call.*3-digit code.*", "IF you receive a phone call with a 3-digit code"),
    (r".*scheduled.*away.*nobody.*house.*", "IF you are scheduled to be away and nobody is in the house"),
    (r".*scheduled.*home.*somebody.*home.*", "IF you are scheduled to be home and somebody is at home"),
    (r".*scheduled.*sleep mode.*somebody.*home.*", "IF you are scheduled to be in sleep mode and somebody is at home"),
    (r".*enter.*room.*", "IF you enter a room"),
    (r".*leave.*room.*", "IF you leave a room"),
    (r".*press.*button.*triby io.*", "IF you press a given button on your Triby IO"),
    (r".*ok google.*block time.*", "IF you say 'ok google' followed by a phrase to block time"),
    (r".*ifttt-command.*xtactor wristband.*", "IF you send an IFTTT command from your Xtactor wristband"),
    (r".*turn on.*alarm.*livy app.*", "IF you turn on the alarm in the Livy app"),
    (r".*abode system mode.*changed.*", "IF your Abode system mode is changed"),
    (r".*air quality.*above or below.*target.*", "IF your air quality goes above or below the target level"),
    (r".*device detects.*humidity.*above.*threshold.*", "IF your device detects humidity above the threshold"),
    (r".*device detects.*temperature.*above.*threshold.*", "IF your device detects temperature above the threshold"),
    (r".*device detects.*temperature.*below.*threshold.*", "IF your device detects temperature below the threshold"),
    (r".*livy protect.*fire or motion alarm.*", "IF your Livy Protect device reports a fire or motion alarm"),
    (r".*local room temperature exceeds.*threshold.*", "IF your local room temperature exceeds a given threshold"),
    (r".*mui board.*turned off night mode.*", "IF your Mui board turned off night mode"),
    (r".*mui board.*turned on night mode.*", "IF your Mui board turned on night mode"),
    (r".*order.*being prepped.*", "IF your order is being prepped"),
    (r".*order.*being quality-checked.*", "IF your order is being quality-checked"),
    (r".*robot completes.*job.*", "IF your robot completes a job"),
    (r".*robot starts.*job.*", "IF your robot starts a job"),
    (r".*smart bed alarm clock.*activated.*", "IF your smart bed alarm clock is activated"),
    (r".*water heater exceeds.*energy use threshold.*", "IF your water heater exceeds its energy use threshold"),
    (r".*motion sensor detects movement.*", "IF a motion sensor detects movement nearby"),
    (r".*posts.*new message.*room.*", "IF someone posts a new message in a selected room"),
    (r".*spend money.*merchant.*transaction.*", "IF you spend money at a particular merchant"),
    (r".*15 minutes.*ending time.*event.*google calendar.*", "IF within 15 minutes of the ending time of a Google Calendar event"),
    (r".*low battery.*detected.*swann.*", "IF low battery is detected on a Swann wireless security device"),
    (r".*enrolled.*face.*detected.*swann.*", "IF an enrolled or selected face is detected by Swann security device"),
    (r".*thermostat.*heating mode.*changes.*", "IF your thermostat's heating mode changes to a certain mode"),
    (r".*thermostat.*set temperature.*changes.*", "IF your thermostat's set temperature changes"),
    (r".*recommends.*new applet.*day.*", "IF a new applet is recommended every day"),
    (r".*child lock.*deactivated.*", "IF the child lock has deactivated"),
    (r".*electricity price falls.*rises.*threshold.*", "IF electricity price falls or rises past your set threshold"),
    (r".*carbon.*clean or dirty.*", "IF the carbon is clean or dirty"),
    (r".*home enters away mode.*", "IF your home enters away mode"),
    (r".*smoke detector.*detects.*smoke alarm.*", "IF your smoke detector detects a smoke alarm"),
    (r".*minutes before.*event.*calendar.*keyword.*", "IF a set number of minutes before a calendar event with a keyword"),
    (r".*run.*tecan instrument.*completed.*", "IF a run on the selected Tecan instrument is completed"),
    (r".*tecan instrument.*state changes.*", "IF the selected Tecan instrument's state changes"),
    (r".*door.*left open.*18 minutes.*", "IF any door is left open for 18 minutes"),
    (r".*alarm panel.*alarm.*", "IF your alarm panel has an alarm"),
    (r".*air quality.*louisville metro.*sensors.*", "IF air quality data is retrieved from Louisville Metro air sensors"),
    (r".*hours.*electricity.*denmark.*cheap.*", "IF at the hours where electricity in Denmark is cheap"),
    (r".*hours.*electricity.*norway.*cheap.*", "IF at the hours where electricity in Norway is cheap"),
    (r".*gust speed rises above.*", "IF the gust speed rises above a specified value"),
    (r".*markets close.*stock.*price drops.*percentage.*", "IF a stock's price drops by a percentage after markets close"),
    (r".*device disconnects.*google wifi.*", "IF a device disconnects from Google WiFi"),
    (r".*air quality alert.*withings home.*", "IF an air quality alert is raised by your Withings Home"),
    (r".*alarm.*detected.*", "IF an alarm is detected"),
    (r".*event matches.*criteria.*", "IF an event matches the specified criteria"),
    (r".*any reminder.*completed.*", "IF any reminder is completed"),
    (r".*someone.*identified.*camera.*arrives home.*", "IF someone already identified by the camera arrives home"),
    (r".*unlocks.*door.*auto unlock.*", "IF someone unlocks the door with auto unlock"),
    (r".*door.*unlocked.*", "IF the door is unlocked"),
    (r".*ifttt blog.*tagged.*updates.*", "IF the IFTTT blog posts something tagged 'updates'"),
    (r".*thermostat.*set to.*heating mode.*", "IF the thermostat is set to a certain heating mode"),
    (r".*ask alexa.*shopping list.*", "IF you ask Alexa what's on your shopping list"),
    (r".*alarm panel.*armed.*", "IF your alarm panel is armed"),
    (r".*alarm panel.*disarmed.*", "IF your alarm panel is disarmed"),
    (r".*check in.*venue.*category.*foursquare.*", "IF you check in to a venue in a specific category on Foursquare"),
    (r".*leave.*voicemail.*ifttt phone number.*", "IF you leave any voicemail at your IFTTT phone number"),
    (r".*orion group.*says.*word or phrase.*", "IF you or someone in your Orion group says a particular word or phrase"),
    (r".*press.*action button.*lametric time.*", "IF you press the action button on LaMetric Time"),
    (r".*receive.*notification.*repository.*github.*", "IF you receive a new notification from a specific repository on GitHub"),
    (r".*receive.*new voicemail.*", "IF you receive a new voicemail"),
    (r".*receive.*incoming call.*not in.*contact list.*", "IF you receive an incoming call from a number not in your contact list"),
    (r".*send.*sms.*android.*matches.*search.*", "IF you send an SMS on your Android device that matches a search"),
    (r".*send.*sms.*android.*", "IF you send an SMS on your Android device"),
    (r".*submit.*new note.*", "IF you submit any new note"),
    (r".*take.*photo.*front camera.*", "IF you take a photo with the front camera"),
    (r".*android.*connects.*wifi.*", "IF your Android device connects to any WiFi network"),
    (r".*cloudbit receives.*input signal.*", "IF your cloudBit receives an input signal from another littleBits module"),
    (r".*d-link.*detects.*motion event.*", "IF your D-Link device detects a motion event"),
    (r".*ecobee.*smart home.*smart away.*", "IF your Ecobee thermostat changes to smart home or smart away comfort profiles"),
    (r".*ge dishwasher.*finishes.*cycle.*", "IF your GE dishwasher finishes a cycle"),
    (r".*ge dryer.*finishes.*cycle.*", "IF your GE dryer finishes a cycle"),
    (r".*lock.*locked.*specific person.*", "IF your lock is locked by a specific person"),
    (r".*myfox.*detects.*intrusion.*", "IF your MyFox security system detects an intrusion"),
    (r".*myfox.*disarmed.*", "IF your MyFox security system is disarmed"),
    (r".*netatmo presence.*detects.*person outside.*", "IF your Netatmo Presence detects a person outside"),
    (r".*netatmo presence.*detects.*vehicle outside.*", "IF your Netatmo Presence detects a vehicle outside"),
    (r".*netatmo rain gauge.*no longer raining.*", "IF your Netatmo rain gauge detects it is no longer raining"),
    (r".*oven.*turned on.*", "IF your oven is turned on"),
    (r".*site changes modes.*", "IF your site changes modes"),
    (r".*smartthings.*temperature.*below.*", "IF your SmartThings device detects temperatures below a specified value"),
    (r".*smartthings.*switched on.*", "IF your SmartThings device is switched on"),
    (r".*weatherflow tempest.*detects.*start.*rain.*", "IF your WeatherFlow Tempest station detects the start of rain"),
    (r".*fourth of july.*", "IF it's the Fourth of July at a specified time"),
]

TRIGGER_RULES += [
    (r".*oven door.*opened.*", "IF your oven door has been opened"),
    (r".*oven turns off.*cooking timer.*zero.*", "IF your oven turns off after the cooking timer reaches zero"),
    (r".*refrigerator door.*opened.*", "IF your refrigerator door is opened"),
    (r".*refrigerator.*freezer doors.*open.*", "IF your refrigerator or freezer doors are open"),
    (r".*roxxter.*started.*", "IF your Roxxter has started"),
    (r".*smartthings.*temperature.*above.*", "IF your SmartThings device detects temperatures above a specified value"),
    (r".*smartthings.*switched off.*", "IF your SmartThings device is switched off"),
    (r".*thermostat.*nearly out.*battery.*", "IF your thermostat is nearly out of battery"),
    (r".*wemo insight.*standby mode.*", "IF your Wemo Insight switch enters standby mode"),
    (r".*wemo insight.*turned off.*", "IF your Wemo Insight switch is turned off"),
    (r".*wemo motion sensor.*detects.*motion.*", "IF your Wemo motion sensor detects new motion after inactivity"),
    (r".*wine cooler door.*opened.*", "IF your wine cooler door has been opened"),
    (r".*iot device.*zone.*action.*", "IF a given combination of IoT device, zone and action occurs"),
    (r".*aura detects motion.*", "IF Aura detects motion"),
    (r".*thermostat encounters.*error.*", "IF the thermostat encounters an error"),
    (r".*blue by adt alarm.*", "IF a Blue by ADT alarm event occurs"),
    (r".*blue by adt sensor.*activity.*home.*", "IF a Blue by ADT sensor notices activity in the home"),
    (r".*contact sensor closes.*", "IF a contact sensor closes"),
    (r".*device.*homeseer.*turned off.*", "IF a device attached to your HomeSeer system is turned off"),
    (r".*door.*open.*garage mode.*", "IF a door is open in garage mode"),
    (r".*door opens.*closes.*moves.*", "IF a door opens, closes, or moves"),
    (r".*parameter exceed.*threshold.*", "IF a given parameter exceeds a threshold"),
    (r".*parameter falls below.*threshold.*", "IF a given parameter falls below a threshold"),
    (r".*human.*detected.*beseye.*", "IF a human is detected by Beseye camera"),
    (r".*knx device.*binary switch.*turned off.*", "IF a KNX device or binary switch is turned off"),
    (r".*knx scene.*started.*", "IF a KNX scene is started"),
    (r".*leak.*detected.*device.*", "IF a leak is detected by a specified device"),
    (r".*mode.*armed.*", "IF a mode is armed"),
    (r".*mode.*disarmed.*", "IF a mode is disarmed"),
    (r".*new deal.*slickdeals.*", "IF a new deal is posted on the frontpage at Slickdeals"),
    (r".*disease outbreak.*world health organization.*", "IF a new disease outbreak update is posted by the WHO"),
    (r".*new iot podcast.*", "IF a new IoT podcast episode is published"),
    (r".*noonlight alarm.*triggered.*", "IF a Noonlight alarm is triggered"),
    (r".*cell.*updated.*spreadsheet.*", "IF a particular cell is updated in a spreadsheet"),
    (r".*person.*detected.*swann.*", "IF a person is detected on your Swann security device"),
    (r".*pet.*detected.*swann.*", "IF a pet is detected on your Swann security device"),
    (r".*sound.*detected.*swann.*", "IF a sound is detected by your Swann security device"),
    (r".*motion sensor.*reed sensor.*door.*window.*open.*", "IF a motion or reed sensor detects a door or window is open"),
    (r".*task.*list.*completed.*", "IF a task in a given list is completed"),
    (r".*vehicle enters.*geo-circle.*", "IF a vehicle enters one of its geo-circles"),
    (r".*vehicle exits.*geo-circle.*", "IF a vehicle exits one of its geo-circles"),
    (r".*vehicle.*detected.*swann.*", "IF a vehicle is detected by your Swann security device"),
    (r".*watering event.*executed.*", "IF a watering event is executed"),
    (r".*wyze cam.*smoke alarm.*", "IF a Wyze Cam detects a smoke alarm"),
    (r".*wyze cam.*detects sound.*", "IF a Wyze Cam detects sound"),
    (r".*adaptors detect intrusion.*", "IF adaptors detect intrusion"),
    (r".*event.*eight alarm.*", "IF an event related to the Eight alarm happens"),
    (r".*interesting event.*device.*particle.*", "IF an interesting event comes from a Particle device"),
    (r".*ohmhour starts.*", "IF an OhmHour starts"),
    (r".*goal.*red.*imminent derailment.*", "IF any goal moves into the red (imminent derailment)"),
    (r".*motion sensor detects motion.*armed.*", "IF any motion sensor detects motion while system is armed"),
    (r".*user sends.*ifttt command.*", "IF any user sends the /ifttt command to the source group"),
    (r".*family member.*come home.*", "IF at least one family member has come home"),
    (r".*away mode.*canceled.*thermostat.*", "IF away mode is canceled on your thermostat"),
    (r".*away mode.*on.*", "IF away mode is on"),
    (r".*bart.*delayed.*station.*", "IF BART is delayed at a station"),
    (r".*battery levels fall.*vehicle.*", "IF battery levels fall below normal for any vehicle"),
    (r".*button.*pressed.*", "IF a button is pressed"),
    (r".*co2 emissions.*low.*", "IF CO2 emissions level is low"),
    (r".*every family member.*left.*", "IF every family member has left"),
    (r".*everyone.*gone to sleep.*", "IF everyone has gone to sleep"),
    (r".*fetch-it button.*pressed.*long pressed.*", "IF Fetch-it button is pressed or long pressed"),
    (r".*home mode.*on.*", "IF home mode is on"),
    (r".*motion.*detected.*camera.*", "IF motion is detected on a selected camera"),
    (r".*motion.*detected.*", "IF motion is detected"),
    (r".*thermostat enters away mode.*", "IF your thermostat enters away mode"),
    (r".*total connect.*alarm.*", "IF your Total Connect security panel receives an alarm"),
    (r".*phyn detects.*alert.*", "IF Phyn detects an alert"),
    (r".*pir detects person.*", "IF PIR detects a person"),
    (r".*simcam detects.*person.*area.*", "IF SimCam detects a person in a specific area"),
    (r".*sound.*detected.*microphone.*", "IF sound is detected on a selected microphone"),
    (r".*air quality.*below.*value.*", "IF the air quality goes below a given value"),
    (r".*alarm.*mode.*dismissed.*", "IF the alarm for a specific mode is dismissed"),
    (r".*alarm.*changed mode.*armed.*disarmed.*", "IF the alarm has changed mode"),
    (r".*alarm.*armed.*", "IF the alarm is armed"),
    (r".*alarm.*disarmed.*", "IF the alarm is disarmed"),
    (r".*aqi.*exceeds.*threshold.*", "IF the AQI of your device exceeds a selected threshold"),
    (r".*co2 intensity.*electricity.*lowest.*", "IF the CO2 intensity in electricity production is lowest in a time period"),
    (r".*cost.*wemo insight.*rises above.*daily.*", "IF the cost of your Wemo Insight switch rises above a daily value"),
    (r".*co₂ level.*lower.*threshold.*", "IF the CO₂ level becomes lower than a specified threshold"),
    (r".*fibaro.*detects motion.*", "IF the device connected to Fibaro Home Center detects motion"),
    (r".*electricity price.*lowest.*", "IF the electricity price is lowest in a time period"),
    (r".*ema.*metrosafe.*louisville.*emergency.*", "IF EMA/MetroSafe in Louisville sends an emergency notification"),
    (r".*energy saving mode.*activated.*", "IF the energy saving mode scene is activated"),
    (r".*filtrete.*air quality.*level changes.*", "IF the Filtrete outdoor air quality scale level changes"),
    (r".*filtrete.*air quality.*higher.*worse.*", "IF the Filtrete outdoor air quality value becomes worse"),
    (r".*gate start opening.*", "IF the gate starts opening"),
    (r".*goal.*close to derailing.*", "IF the goal is close to derailing"),
    (r".*home security.*enters away mode.*", "IF the home security device enters away mode"),
    (r".*hub.*specific state.*", "IF the hub is in a specific state"),
    (r".*humidity.*rises above.*threshold.*", "IF the humidity rises above a provided threshold"),
    (r".*smartphone screen unlocks.*exceeds.*daily limit.*", "IF your smartphone screen unlocks exceed your daily limit"),
    (r".*outdoor temperature rises above.*", "IF the outdoor temperature rises above a given value"),
    (r".*filtrete.*air filter.*remaining life.*below.*", "IF the remaining life of your Filtrete air filter drops below a percentage"),
    (r".*room temperature falls below.*", "IF the room temperature falls below a user-defined threshold"),
    (r".*room temperature rises above.*", "IF the room temperature rises above a user-defined value"),
    (r".*swann.*detects.*motion.*", "IF your Swann security device detects motion"),
    (r".*smart alert.*smoke.*carbon monoxide.*", "IF the smart alert detects smoke or carbon monoxide alarm"),
    (r".*camera detects motion.*", "IF a specified camera detects motion"),
    (r".*device.*set to cool mode.*", "IF a specified device is set to cool mode"),
]

TRIGGER_RULES += [
    (r".*contact sensor opens.*", "IF a contact sensor opens"),
    (r".*user.*detected.*nearby.*network.*", "IF a user is detected in or nearby your network"),
    (r".*user sends.*ifttt command.*key phrase.*", "IF a user sends a message with the /ifttt command and a key phrase"),
    (r".*alarm event occurs.*", "IF an alarm event occurs"),
    (r".*alarm sounds.*", "IF an alarm sounds"),
    (r".*issue occurs.*low battery.*device offline.*", "IF an issue occurs, such as low battery or device offline"),
    (r".*door panel.*opened.*", "IF any or a specific door panel is opened"),
    (r".*caavo.*search query.*", "IF Caavo Control Center performs any search query"),
    (r".*global air pollution.*higher than 50.*", "IF global air pollution is higher than 50"),
    (r".*lock.*unlocked.*", "IF lock is unlocked"),
    (r".*no one.*user group.*detected.*network.*", "IF no one of a user group is detected in or nearby your network"),
    (r".*simcam recognizes.*face id.*", "IF SimCam recognizes a face ID that is set up"),
    (r".*pir kumosensor detects movement.*", "IF specified PIR KumoSensor detects movement"),
    (r".*air quality.*air monitor.*unhealthy.*", "IF the air quality measured from your air monitor is unhealthy"),
    (r".*aqi.*device.*updated.*", "IF the AQI of your device is updated"),
    (r".*co₂ level.*exceeded.*threshold.*", "IF the CO₂ level for a specific device exceeds a threshold"),
    (r".*daily rain measurement rises above.*", "IF the daily rain measurement rises above a given threshold"),
    (r".*garage door starts to open.*", "IF the garage door starts to open"),
    (r".*radon level.*rises above.*threshold.*", "IF the radon level for a device rises above the provided threshold"),
    (r".*smarter coffee.*finished brewing.*", "IF the Smarter Coffee 2.0 has finished brewing"),
    (r".*smarter ikettle.*finished boiling.*", "IF the Smarter iKettle 3.0 has finished boiling"),
    (r".*uv index rises above.*", "IF the UV index rises above a given value"),
    (r".*time-of-day peak rates.*start or end.*", "IF time-of-day peak rates start or end"),
    (r".*text message.*key phrase.*ifttt bot.*telegram.*", "IF you send a text message with a key phrase to the @ifttt bot on Telegram"),
    (r".*tap.*night mode.*day mode.*eight app.*", "IF you tap the 'night mode' or 'day mode' buttons in the Eight app"),
    (r".*killed in rust.*not playing.*", "IF you were killed in Rust while not playing the game"),
    (r".*ewelink.*1-channel switch.*turned on or off.*", "IF your eWeLink 1-channel switch is turned on or off"),
    (r".*order.*in the oven.*", "IF your order is in the oven"),
    (r".*order.*ready for pickup.*", "IF your order is ready for pickup"),
    (r".*smanos system.*armed.*away mode.*", "IF your Smanos system is armed to away mode with no one in the house"),
    (r".*carbon intensity changes.*comed.*", "IF the carbon intensity changes for the ComEd service territory"),
    (r".*press.*button.*wink relay.*", "IF you press the selected button on your Wink Relay"),
    (r".*assigned pusher.*festina watch.*", "IF the assigned pusher on your Festina watch is fired"),
    (r".*assigned pusher.*jaguar watch.*", "IF the assigned pusher on your Jaguar watch is fired"),
    (r".*assigned pusher.*lotus watch.*", "IF the assigned pusher on your Lotus watch is fired"),
    (r".*thermostat.*measured temperature.*exceeds.*threshold.*", "IF your thermostat's measured temperature exceeds a certain threshold"),
    (r".*thermostat.*set temperature.*exceeds.*threshold.*", "IF your thermostat's set temperature exceeds a certain threshold"),
    (r".*local uv levels.*uv index rises above.*", "IF your local UV index rises above a specified value"),
    (r".*door.*locked.*", "IF the door is locked"),
    (r".*swann video doorbell button.*pressed.*", "IF the Swann video doorbell button is pressed"),
    (r".*stock.*closing price rises above.*", "IF a stock's closing price rises above a specified amount"),
    (r".*new reminder.*added.*list.*", "IF a new reminder is added to a specified list"),
    (r".*specific person arrives home.*", "IF a specific person arrives home"),
    (r".*alarm.*triggered.*system.*", "IF an alarm is triggered on your system"),
    (r".*new notification.*android.*", "IF any new notification is received on your Android device"),
    (r".*smoke.*detected.*", "IF smoke is detected"),
    (r".*camera detects.*sound or motion.*", "IF the camera detects a sound or motion event"),
    (r".*clock timer.*cooktop.*elapsed.*", "IF the clock timer on your cooktop has elapsed"),
    (r".*clock timer.*oven.*elapsed.*", "IF the clock timer on your oven has elapsed"),
    (r".*laundry cycle ends.*", "IF the laundry cycle ends"),
    (r".*netatmo welcome.*detects.*person.*identified.*", "IF Netatmo Welcome detects a person that has already been identified"),
    (r".*noise rises above.*decibel.*", "IF the noise rises above a decibel value you specify"),
    (r".*temperature drops below.*threshold.*", "IF the temperature drops below a threshold you specify"),
    (r".*temperature rises above.*threshold.*", "IF the temperature rises above a threshold you specify"),
    (r".*achieve.*daily distance goal.*", "IF you achieve your daily distance goal"),
    (r".*click.*button.*thinga.*", "IF you click a button on Thinga"),
    (r".*double press.*misfit flash button.*", "IF you double press your Misfit Flash button"),
    (r".*enter or exit.*area.*", "IF you enter or exit a specified area"),
    (r".*like.*tweet.*", "IF you like a tweet"),
    (r".*log.*weight.*fitbit.*aria.*", "IF you log your weight in the Fitbit app or with the Aria scale"),
    (r".*post.*new photo.*facebook.*", "IF you post a new photo on Facebook"),
    (r".*post.*new tweet.*", "IF you post a new tweet"),
    (r".*press the stone.*", "IF you press the Stone"),
    (r".*receive.*incoming call.*contact list.*", "IF you receive an incoming call from a phone number in your contact list"),
    (r".*receive.*sms.*android.*phone number.*", "IF you receive an SMS on your Android device from a specified phone number"),
    (r".*receive.*sms.*android.*matches.*search.*", "IF you receive an SMS on your Android device that matches a search"),
    (r".*upload.*new public video.*youtube.*", "IF you upload a new public video to YouTube"),
    (r".*cookit.*finished successfully.*", "IF your Cookit has finished successfully"),
    (r".*dishwasher cycle.*complete.*", "IF your dishwasher cycle is complete"),
    (r".*dryer cycle.*complete.*wrinkle guard.*", "IF your dryer cycle is complete"),
    (r".*lock.*unlocked.*specific person.*", "IF your lock is unlocked by a specific person"),
    (r".*smartthings.*presence.*no longer detected.*", "IF your SmartThings device's presence is no longer detected"),
    (r".*wemo insight.*turned on.*", "IF your Wemo Insight switch is turned on"),
    (r".*wemo light.*switches off.*", "IF your Wemo light switches off"),
    (r".*dynamic pricing.*average price changes.*", "IF the average price changes on a dynamic pricing plan"),
    (r".*door.*closed.*", "IF a door is closed"),
    (r".*nexia automation runs.*", "IF a Nexia automation runs"),
    (r".*scheduled meeting ends.*", "IF a scheduled meeting ends"),
    (r".*scheduled meeting starts.*", "IF a scheduled meeting starts"),
    (r".*task.*completed.*", "IF a task is completed"),
    (r".*time for fast.*", "IF it's time for fast"),
    (r".*alarm.*triggered.*", "IF the alarm is triggered"),
    (r".*renewable generation.*grid changes.*", "IF the amount of renewable generation in the grid changes"),
    (r".*price of electricity changes.*", "IF the price of electricity changes"),
    (r".*ok google.*set nest thermostat.*", "IF you say 'ok google' followed by a phrase to set Nest thermostat"),
    (r".*robot finishes cleaning.*room.*", "IF your robot finishes cleaning a room"),
    (r".*robot starts cleaning.*room.*", "IF your robot starts cleaning a room"),
    (r".*assigned pusher.*kronaby watch.*", "IF the assigned pusher on your Kronaby watch is fired"),
    (r".*ubibot.*light sensor.*light level.*lux.*", "IF the UbiBot device's light level rises above or drops below a value in lux"),
    (r".*local pollen levels.*pollen count rises.*", "IF your local pollen count rises above a specified value"),
    (r".*tomorrow.*weather report.*", "IF tomorrow's weather report is retrieved"),
    (r".*door.*unlocked.*", "IF the door is unlocked"),
    (r".*tags.*flics.*", "IF you create tags for your Flics"),
    (r".*stock.*price rises.*percentage.*", "IF a stock's price rises by a percentage after markets close"),
    (r".*astronaut enters space.*", "IF an astronaut enters space"),
    (r".*netatmo welcome.*detects.*unknown person.*", "IF Netatmo Welcome detects an unknown person"),
    (r".*preheating phase.*finished.*fast preheat.*", "IF the preheating phase has finished with fast preheat enabled"),
    (r".*complete.*item.*to do list.*", "IF you complete an item on your to-do list"),
    (r".*like.*video.*youtube.*", "IF you like a video on YouTube"),
    (r".*log sleep above.*target hours.*", "IF you log sleep above the target hours you specify"),
]

TRIGGER_RULES += [
    (r".*receive.*incoming call.*", "IF you receive an incoming call"),
    (r".*ifttt tag.*mesh app.*input signal.*", "IF your IFTTT tag on Mesh app receives an input signal"),
    (r".*smartthings.*presence.*detected.*", "IF your SmartThings device's presence is detected"),
    (r".*wemo light.*switches on.*", "IF your Wemo light switches on"),
    (r".*door.*opened.*", "IF a door is opened"),
    (r".*motion sensor.*clear.*", "IF a motion sensor becomes clear"),
    (r".*routine.*activated.*", "IF a routine is activated"),
    (r".*event.*triggered.*litter robot.*", "IF an event is triggered from your Litter Robot"),
    (r".*tado.*switches.*home mode.*", "IF tado° switches to home mode"),
    (r".*alarm.*specific mode.*triggered.*", "IF the alarm for a specific mode is triggered"),
    (r".*scene.*room.*changed.*", "IF the scene in a room is changed"),
    (r".*device connects.*google wifi.*", "IF a device connects to Google WiFi"),
    (r".*new item.*feed.*keyword.*phrase.*", "IF a new item in a feed contains a particular keyword or phrase"),
    (r".*specified camera.*detects motion.*", "IF a specified camera detects motion"),
    (r".*stream.*going live.*channel.*", "IF a stream is going live for a specified channel you follow"),
    (r".*carbon dioxide drops below.*", "IF the carbon dioxide drops below a specified value"),
    (r".*specified lock.*unlocked.*", "IF the specified lock is unlocked"),
    (r".*temperature drops below.*value.*", "IF the temperature drops below a value you specify"),
    (r".*myfox.*receives.*alarm.*detection.*intrusion.*", "IF your MyFox security system receives an alarm"),
    (r".*oven.*finished successfully.*", "IF your oven has finished successfully"),
    (r".*smartthings device.*opened.*", "IF your SmartThings device is opened"),
    (r".*washer cycle.*complete.*", "IF your washer cycle is complete"),
    (r".*wemo switch.*turned on.*", "IF your Wemo switch is turned on"),
    (r".*withings sleep.*get into bed.*", "IF your Withings Sleep detects that you get into bed"),
    (r".*withings sleep.*get out of bed.*", "IF your Withings Sleep detects that you get out of bed"),
    (r".*motion sensor detects motion.*", "IF a motion sensor detects motion"),
    (r".*tado.*switches.*away mode.*", "IF tado° switches to away mode"),
    (r".*during alarms.*", "IF during alarms"),
    (r".*new photo.*added.*camera roll.*", "IF a new photo is added to your camera roll"),
    (r".*new track.*added.*playlist.*", "IF a new track is added to a playlist you specify"),
    (r".*new tweet matches.*search query.*", "IF a new tweet matches your search query"),
    (r".*new user.*following you.*", "IF a new user starts following you"),
    (r".*nest protect.*warning carbon monoxide.*", "IF Nest Protect detects warning carbon monoxide levels"),
    (r".*temperature rises above.*value.*", "IF the temperature rises above a value you specify"),
    (r".*place.*phone call.*android.*", "IF you place a phone call on your Android device"),
    (r".*post.*new tweet.*hashtag.*", "IF you post a new tweet with a specific hashtag"),
    (r".*netatmo rain gauge.*raining.*", "IF your Netatmo rain gauge detects that it is raining"),
    (r".*skybell hd.*detects motions.*", "IF your SkyBell HD detects motions"),
    (r".*carbon dioxide.*co₂.*awair.*rises above.*", "IF the CO₂ concentration from your Awair device rises above the threshold"),
    (r".*new super sticker.*live chat.*", "IF there is a new super sticker in a live chat during a live stream"),
    (r".*order.*out for delivery.*", "IF your order is out for delivery"),
    (r".*achieve.*daily step goal.*", "IF you achieve your daily step goal"),
    (r".*share.*new photo.*instagram.*", "IF you share any new photo on Instagram"),
    (r".*android device.*plugged in.*", "IF your Android device is plugged in"),
    (r".*android.*battery drops below 15%.*", "IF your Android device's battery drops below 15%"),
    (r".*smartthings.*senses motion.*", "IF your SmartThings device senses motion"),
    (r".*wemo light switch.*turned off.*", "IF your Wemo light switch is turned off"),
    (r".*wyze cam.*detects motion.*", "IF a Wyze Cam detects motion"),
    (r".*abode alarm.*active.*", "IF an Abode alarm is active"),
    (r".*blink camera.*detects motion.*", "IF your Blink camera detects motion"),
    (r".*plage tarifaire.*commence.*logement.*", "IF the chosen pricing range starts in your home"),
    (r".*minutes before.*starting time.*event.*google calendar.*", "IF a set number of minutes before any Google Calendar event"),
    (r".*christmas.*time zone.*", "IF it is Christmas in a specified time zone"),
    (r".*add.*new reminder.*", "IF you add a new reminder"),
    (r".*miss.*phone call.*android.*number.*", "IF you miss a phone call on your Android device from a specified number"),
    (r".*android.*disconnects.*wifi.*specify.*", "IF your Android device disconnects from a specified WiFi network"),
    (r".*button.*pressed.*skybell hd.*", "IF the button is pressed on your SkyBell HD"),
    (r".*twitter user.*tweets.*", "IF the Twitter user you specify tweets"),
    (r".*ask alexa.*to do list.*", "IF you ask Alexa what's on your to-do list"),
    (r".*fitbit logs.*new sleep.*", "IF your Fitbit logs new sleep"),
    (r".*wemo switch.*turned off.*", "IF your Wemo switch is turned off"),
    (r".*withings home.*detects motion.*", "IF your Withings Home detects motion"),
    (r".*ihome enhance button.*pressed.*", "IF an iHome Enhance button is pressed"),
    (r".*local temperature.*rises above.*value.*", "IF your local temperature rises above a specified value"),
    (r".*minutes before.*event.*google calendar.*timezone.*", "IF a set number of minutes before a Google Calendar event"),
    (r".*audio event.*detected.*", "IF an audio event is detected"),
    (r".*home.*set to home.*", "IF the home you specify is set to home"),
    (r".*international space station.*passes over.*location.*", "IF the International Space Station passes over a location you specify"),
    (r".*new follower.*channel.*", "IF there is a new follower of your channel"),
    (r".*save.*new track.*music.*spotify.*", "IF you save a new track to your music on Spotify"),
    (r".*tomorrow.*forecasted weather condition.*", "IF tomorrow's forecasted weather condition changes"),
    (r".*camera detects.*motion event.*", "IF the camera detects a motion event"),
    (r".*carbon dioxide rises above.*", "IF the carbon dioxide rises above a specified value"),
    (r".*nest protect.*dangerous carbon monoxide.*", "IF Nest Protect detects dangerous carbon monoxide levels"),
    (r".*@mentioned.*tweet.*", "IF you are @mentioned in a tweet"),
    (r".*android.*connects.*wifi.*specify.*", "IF your Android device connects to a specified WiFi network"),
    (r".*wemo light switch.*turned on.*", "IF your Wemo light switch is turned on"),
    (r".*wemo motion sensor.*detects.*new motion.*", "IF your Wemo motion sensor detects new motion"),
    (r".*send.*email.*trigger@applet.*ifttt.*", "IF you send IFTTT any email at trigger@applet.ifttt.com"),
    (r".*motion event.*detected.*", "IF a motion event is detected"),
    (r".*new item.*added.*feed.*", "IF a new item is added to a feed you specify"),
    (r".*song.*played.*amazon prime music.*", "IF a song is played on Amazon Prime Music"),
    (r".*email.*trigger@applet.*hashtag.*subject.*", "IF you send IFTTT an email with a hashtag in the subject"),
    (r".*add.*item.*to do list.*", "IF you add an item to your to-do list"),
    (r".*time for prayer.*", "IF it's time for prayer"),
    (r".*local temperature.*drops below.*value.*", "IF your local temperature drops below a specified value"),
    (r".*minutes before.*event.*calendar.*keyword.*phrase.*", "IF a set number of minutes before a calendar event with a keyword or phrase"),
    (r".*new notification.*android.*app.*specify.*", "IF a new notification is received on your Android device from a specified app"),
    (r".*nest protect.*warning smoke.*", "IF Nest Protect detects warning smoke levels"),
    (r".*tap.*gesture.*custom ifttt task.*", "IF you tap a gesture with a custom IFTTT task"),
    (r".*email.*trigger@applet.*hashtag.*subject.*attachment.*", "IF you send IFTTT an email with a hashtag and optional attachment"),
    (r".*miss.*phone call.*android.*", "IF you miss a phone call on your Android device"),
    (r".*ok google.*post.*tweet.*", "IF you say 'ok google' followed by a phrase to post a tweet"),
    (r".*email.*trigger@applet.*ifttt.*attachment.*", "IF you send IFTTT any email with optional attachment"),
    (r".*home.*set to away.*", "IF the home you specify is set to away"),
    (r".*create.*plain text status.*facebook.*", "IF you create a new plain text status message on Facebook"),
    (r".*specific user.*subscribed.*new video public.*", "IF a specific user you're subscribed to makes a new video public"),
    (r".*nest protect.*dangerous smoke.*", "IF Nest Protect detects dangerous smoke levels"),
    (r".*receive.*new sms.*android.*", "IF you receive any new SMS on your Android device"),
    (r".*click.*selected flic.*", "IF you click the selected Flic"),
    (r".*wemo light switch.*long press.*two seconds.*", "IF your Wemo light switch is turned on or off with a long press"),
    (r".*ifttt receives.*event.*apilio.*", "IF IFTTT receives a specific event from Apilio"),
    (r".*new super chat.*live chat.*", "IF there is a new super chat message in a live chat during a live stream"),
    (r".*today.*current weather report.*", "IF today's current weather report is retrieved"),
    (r".*new membership.*channel.*", "IF there is a new membership for your channel"),
    (r".*motion.*detected.*doorbell.*", "IF motion is detected at a given doorbell"),
    (r".*timer goes off.*", "IF your timer goes off"),
    (r".*rings.*ring doorbell.*", "IF somebody rings your Ring doorbell"),
    (r".*tagged.*new photo.*", "IF you are tagged in a new photo"),
    (r".*once.*hour.*:00.*:15.*:30.*:45.*", "IF once an hour at :00, :15, :30, or :45 minutes past the hour"),
    (r".*arlo.*detects motion.*", "IF your Arlo device detects motion"),
    (r".*alarm goes off.*", "IF your alarm goes off"),
    (r".*exit.*area.*specify.*", "IF you exit an area you specify"),
    (r".*once.*year.*date.*time.*", "IF once a year on a specified date and time"),
    (r".*15 minutes.*sunrise.*location.*", "IF within 15 minutes of sunrise in your location"),
    (r".*current weather condition.*rain.*snow.*cloudy.*clear.*", "IF the current weather condition changes"),
    (r".*enter.*area.*specify.*", "IF you enter an area you specify"),
    (r".*press the button.*", "IF you press the button"),
    (r".*specific days.*week.*time.*", "IF only on specific days of the week at a specified time"),
    (r".*15 minutes.*sunset.*location.*", "IF within 15 minutes of sunset in your location"),
    (r".*ok google.*phrase.*choose.*", "IF you say 'ok google' followed by a phrase you choose"),
    (r".*every single day.*specific time.*", "IF every single day at a specific time"),
    (r".*alexa trigger.*phrase.*defined.*", "IF you say 'Alexa trigger' plus the phrase you have defined"),
    (r".*someone unknown rings.*doorbell.*opener.*", "IF someone unknown rings the doorbell connected to the Opener"),
    (r".*30 minutes.*end.*wash cycle.*", "IF 30 minutes after the end of a wash cycle"),
    (r".*30 minutes.*end.*dryer cycle.*", "IF 30 minutes after the end of a dryer cycle"),
    (r".*check in.*foursquare.*swarm.*hashtag.*shout.*", "IF you check in on Foursquare or Swarm with a single hashtag in the shout"),
    (r".*thermostat.*heating mode.*changes.*set to.*mode.*", "IF your thermostat's heating mode changes to a certain mode"),
    (r".*remaining life.*filtrete.*air filter.*drops below.*percentage.*", "IF the remaining life of your Filtrete Smart Air Filter drops below a chosen percentage"),
    (r".*rule.*sighthound video.*camera.*sends.*event.*", "IF a rule on a Sighthound Video camera sends an event"),
    (r".*changes in your thermostat.*heating mode.*", "IF the thermostat changes heating mode")
]

ACTION_RULES = [
    (r".*change.*brightness.*wemo.*group.*|.*change.*brightness.*wemo.*light.*",", THEN sets the brightness level of smart lights [ACTION_Z]."),
    (r".*sleep fader.*wemo.*group.*|.*sleep fader.*wemo.*light.*", ", THEN start smart lights [ACTION_Z]."),
    (r".*turns off.*light switch.*|.*turn off.*light switch.*", ", THEN turns off a light switch [ACTION_Y]."),
    (r".*turns on.*light switch.*|.*turn on.*light switch.*", ", THEN turns on a light switch in [ACTION_X]."),
    (r".*set.*scene.*hue.*|.*activate.*scene.*|.*nanoleaf.*scene.*|.*wiz.*scene.*|.*lutron.*scene.*", ", THEN activates a lighting scene [ACTION_Y]."),
    (r".*briefly.*hue.*off.*on.*", ", THEN briefly toggles the lights off and on [ACTION_Z]."),
    (r".*change.*brightness.*bulb.*|.*set.*brightness.*light.*|.*dim.*brighten.*", ", THEN adjusts the brightness of lights [ACTION_W]."),
    (r".*change.*color temperature.*|.*specified color.*|.*bulb.*", ", THEN sets the color temperature of lights [ACTION_Y]."),
    (r".*change.*color.*nanoleaf.*|.*change.*color.*hue.*|.*change.*color.*lights.*|.*set.*color.*", ", THEN changes the color of lights [ACTION_Y]."),
    (r".*random.*color.*|.*match.*dominant colors.*", ", THEN sets lights to a random or image-based color [ACTION_Y]."),
    (r".*dynamic mode.*|.*flashing.*|.*jumping.*|.*strobe.*", ", THEN enables [ACTION_X] as lighting mode."),
    (r".*pulse.*gently.*|.*blink.*|.*breathe.*", ", THEN triggers a light animation effect [ACTION_Z]."),
    (r".*pre-defined scene.*", ", THEN sets lights to a predefined scene [ACTION_Z]."),
    (r".*switch off.*light.*", ", THEN switches off the lights [ACTION_W]."),
    (r".*switch on.*light.*", ", THEN switches on the lights [ACTION_Z]."),
    (r".*temporarily.*nanoleaf.*solid.*blinking.*", ", THEN changes light color [ACTION_Y]."),
    (r".*toggle.*wemo.*|.*toggle.*hive.*|.*toggle.*hue.*|.*toggle.*lights.*", ", THEN toggles the lights on or off [ACTION_Y]."),
    (r".*turn.*wemo.*off.*on.*|.*turn.*wemo.*on.*off.*", ", THEN power-cycles a smart switch in [ACTION_X]."),
    (r".*turn off.*functional light.*hood.*|.*turn off.*ambient light.*hood.*", ", THEN turns off the hood lighting [ACTION_W]."),
    (r".*turn on.*ambient light.*hood.*|.*turn on.*functional light.*hood.*", ", THEN turns on the hood lighting [ACTION_Z]."),
    (r".*turn off.*nanoleaf.*|.*turn off.*hue.*|.*turn off.*sengled.*|.*turn off.*selected light.*", ", THEN turns off smart lights [ACTION_W]."),
    (r".*turn on.*nanoleaf.*|.*turn on.*hue.*|.*turn on.*sengled.*|.*turn on.*selected light.*", ", THEN turns on smart lights in [ACTION_X]."),
    (r".*color loop.*hue.*", ", THEN starts a color loop lighting effect [ACTION_Y]."),
    (r".*on/off.*led.*projector.*", ", THEN toggles the projector LED light [ACTION_Z]."),
    (r".*night light.*off.*|.*night light.*on.*|.*hive.*off.*duration.*", ", THEN turns off the light [ACTION_W]."),
    (r".*hive.*on.*duration.*", ", THEN turns on the light [ACTION_Y]."),
    (r".*turn your lights off if.*on.*|.*turn your lights on if.*off.*", ", THEN toggles lights based on current state in [ACTION_X]."),
    (r".*turn your lights off.*|.*turns off.*wemo.*group.*|.*turns off.*wemo.*light.*", ", THEN turns off smart lights [ACTION_W]."),
    (r".*turn your lights on.*|.*turns on.*wemo.*group.*|.*turns on.*wemo.*light.*", ", THEN turns on smart lights [ACTION_X]."),
    (r".*adjust.*multiple.*lights.*shades.*activating.*scene.*lutron.*|.*lutron.*app.*scene.*", ", THEN activates a lighting scene in [ACTION_X]."),
    (r".*turn.*wemo.*light switch.*on.*remain on.*|.*turn.*wemo.*switch.*on.*", ", THEN turns on a light switch [ACTION_Y]."),
    (r".*turn.*wemo.*light switch.*off.*remain off.*|.*turn.*wemo.*switch.*off.*", ", THEN turns off a light switch [ACTION_W]."),
    (r".*set.*(your )?lights?.*specified brightness.*|.*set.*dim level.*light.*|"
     r".*set.*light.*specified brightness.*|.*change.*light.*brightness.*specified level.*|.*change.*lights?.*brightness.*level.*",
     ", THEN adjusts the brightness of lights [ACTION_Z].")
]

# ===============================
# NORMALIZATION FUNCTION
# ===============================


def normalize(df):

    unmatched_triggers = []
    unmatched_actions = []

    def normalize_text(text, RULES, unmatched_list):
        if not isinstance(text, str):
            return text

        text_l = text.lower()

        for pattern, replacement in RULES:
            if re.search(pattern, text_l):
                return replacement

        unmatched_list.append(text)
        return text


    df["triggerDesc"] = df["triggerDesc"].apply(
        normalize_text,
        args=(TRIGGER_RULES, unmatched_triggers)
    )


    df["actionDesc"] = df["actionDesc"].apply(
        normalize_text,
        args=(ACTION_RULES, unmatched_actions)
    )


    return unmatched_triggers, unmatched_actions


# ===============================
# PROCESS
# ===============================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

filtered_df = df

filtered_df = filtered_df[
    ~filtered_df["triggerDesc"].str.contains(
        r".*space.*|.*astronaut.*|.*applet every day.*|.*ce trigger.*|.*health organization.*|"
        r".*tecan instrument.*|.*frontpage at slickdeals.*|.*swann.*|.*Withings scale.*|"
        r".*wordpress.*|.*wemo insight.*|.*facebook.*location.*|.*to do list.*|.*vimeo.*|.*Facebook.*|.*Pandora.*|"
        r".*withings.*weight.*|.*in the livy.*|.*20 items.*|.*WeMo Insight Switch rises.*|.*eight alarm.*",
        case=False,
        na=False
    )
]

filtered_df = filtered_df[
    ~filtered_df["actionDesc"].str.contains(
        r".*wiz lights.*",
        case=False,
        na=False
    )
]

filtered_df.to_csv(INPUT_FILE, index=False)

print(f"NUMBER OF ROWS BEFORE FILTERING: {len(df)}")
print(f"NUMBER OF ROWS AFTER FILTERING: {len(filtered_df)}")

df["originalTrigger"] = df["triggerDesc"]
df["originalAction"] = df["actionDesc"]

unmatched_triggers = []
unmatched_actions = []
unmatched_triggers, unmatched_actions = normalize(df)

# ===============================
# SAVE OUTPUT
# ===============================

folder = os.path.dirname(INPUT_FILE)
base = os.path.basename(INPUT_FILE).replace(".csv", "")

OUTPUT_FILE = os.path.join(
    folder,
    f"{base}_normalized_for_oversampling.csv"
)

df.to_csv(OUTPUT_FILE, index=False)

print(f"SAVED: {OUTPUT_FILE}")

# ===============================
# VALIDATION / REPORT
# ===============================

print("\n========== NORMALIZATION REPORT ==========\n")

print("Loading dataset...")

df = pd.read_csv(OUTPUT_FILE)

unique_triggers = df["triggerDesc"].dropna().unique()
unique_actions = df["actionDesc"].dropna().unique()

print(f"UNIQUE TRIGGERS NORMALIZED: {unique_triggers.__len__()}")
print(f"UNIQUE ACTIONS NORMALIZED: {unique_actions.__len__()}")

# Trigger stats
unique_triggers = df["originalTrigger"].dropna().unique()
unique_unmatched_triggers = list(set(unmatched_triggers))

total_unique_triggers = len(unique_triggers)
unmatched_unique_triggers = len(unique_unmatched_triggers)

unique_trigger_coverage = 100 * (total_unique_triggers - unmatched_unique_triggers) / total_unique_triggers

# Action stats
unique_actions = df["originalAction"].dropna().unique()
unique_unmatched_actions = list(set(unmatched_actions))

total_unique_actions = len(unique_actions)
unmatched_unique_actions = len(unique_unmatched_actions)

unique_action_coverage = 100 * (total_unique_actions - unmatched_unique_actions) / total_unique_actions

print("\n========== UNIQUE COVERAGE ==========\n")

print(f"UNIQUE TRIGGERS NORMALIZED: "
      f"{total_unique_triggers - unmatched_unique_triggers}/{total_unique_triggers} "
      f"({unique_trigger_coverage:.2f}%)")

print(f"UNIQUE ACTIONS NORMALIZED: "
      f"{total_unique_actions - unmatched_unique_actions}/{total_unique_actions} "
      f"({unique_action_coverage:.2f}%)")

print("\n========== ALL UNMATCHED TRIGGERS ==========\n")

if unique_unmatched_triggers.__len__() == 0:
    print("No unmatched triggers")
else:
    print(unique_unmatched_triggers)

print("\n========== ALL UNMATCHED ACTIONS ==========\n")

if unique_unmatched_actions.__len__() == 0:
    print("No unmatched actions")
else:
    print(unique_unmatched_actions)

print("==========================================")