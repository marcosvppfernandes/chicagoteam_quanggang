import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('seaborn')
sns.set_theme(style="whitegrid")
sns.set_color_codes("muted")


def crashes_by_month(crashes):

    crashes['crash_month'] = crashes['crash_date'].apply(lambda x: int(x[:2]))

    crash_no_inj = crashes[crashes['injuries_total'] == 0]
    crash_inj = crashes[crashes['injuries_total'] > 0]

    crash_no_inj = crash_no_inj.groupby('crash_month').sum()
    crash_inj = crash_inj.groupby('crash_month').sum()

    cni = crash_no_inj['num_units']
    cyi = crash_inj['num_units']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(x=cni.index, y="num_units", data=cni,
                label="No Injury Accidents", color="m")

    sns.barplot(x=cyi.index, y="num_units", data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Month")
    month_ints = [i for i in range(1, 13)]
    months = [datetime.date(1900, monthint, 1).strftime('%B')
              for monthint in month_ints]
    ax.set_xticklabels(months, rotation=45)
    sns.despine(left=True, bottom=True)


def crashes_by_hour(crashes):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    import pandas as pd
    crash_no_inj = crashes[crashes['injuries_total'] == 0]
    crash_inj = crashes[crashes['injuries_total'] > 0]

    crash_no_inj = crash_no_inj.groupby('crash_hour').sum()
    crash_inj = crash_inj.groupby('crash_hour').sum()

    cni = crash_no_inj['num_units']
    cyi = crash_inj['num_units']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x=cni.index, y="num_units", data=cni,
                label="No Injury Accidents", color="m")

    sns.set_color_codes("muted")
    sns.barplot(x=cyi.index, y="num_units", data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Hour")
    sns.despine(left=True, bottom=True)


def crashes_by_day(crashes):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    import pandas as pd
    import calendar
    crash_no_inj = crashes[crashes['injuries_total'] == 0]
    crash_inj = crashes[crashes['injuries_total'] > 0]

    crash_no_inj = crash_no_inj.groupby('crash_day_of_week').sum()
    crash_inj = crash_inj.groupby('crash_day_of_week').sum()

    cni = crash_no_inj['num_units']
    cyi = crash_inj['num_units']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x=cni.index, y="num_units", data=cni,
                label="No Injury Accidents", color="m")

    sns.set_color_codes("muted")
    sns.barplot(x=cyi.index, y="num_units", data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="upper right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Day of Week")
    sns.despine(left=True, bottom=True)
    xticks = list(calendar.day_name)[-1:]+list(calendar.day_name)[:6]
    ax.set_xticklabels(xticks, rotation=45)
    sns.set(rc={'axes.facecolor': 'white', 'figure.facecolor': 'white'})


def crashes_by_age(people):

    drivers = people[people['person_type'] == 'DRIVER']

    conds = [((drivers['age'] >= 16) & (drivers['age'] <= 18)),
             ((drivers['age'] >= 19) & (drivers['age'] <= 21)),
             ((drivers['age'] >= 22) & (drivers['age'] <= 24)),
             ((drivers['age'] >= 25) & (drivers['age'] <= 27)),
             ((drivers['age'] >= 28) & (drivers['age'] <= 30)),
             ((drivers['age'] >= 31) & (drivers['age'] <= 33)),
             ((drivers['age'] >= 34) & (drivers['age'] <= 36)),
             ((drivers['age'] >= 37) & (drivers['age'] <= 39)),
             ((drivers['age'] >= 40) & (drivers['age'] <= 42)),
             ((drivers['age'] >= 43) & (drivers['age'] <= 45)),
             ((drivers['age'] >= 46) & (drivers['age'] <= 48)),
             ((drivers['age'] >= 49) & (drivers['age'] <= 51)),
             ((drivers['age'] >= 52) & (drivers['age'] <= 54)),
             ((drivers['age'] >= 55) & (drivers['age'] <= 57)),
             ((drivers['age'] >= 58) & (drivers['age'] <= 60)),
             ((drivers['age'] >= 61) & (drivers['age'] <= 63)),
             ((drivers['age'] >= 64) & (drivers['age'] <= 66)),
             ((drivers['age'] >= 67) & (drivers['age'] <= 69)),
             ((drivers['age'] >= 70) & (drivers['age'] <= 72)),
             (drivers['age'] > 73)]
    choices = ['16-18', '19-21', '22-24', '25-27', '28-30', '31-33',
               '34-36', '37-39', '40-42', '43-45', '46-48', '49-51',
               '52-54', '55-57', '58-60', '61-63', '64-66', '67-69',
               '70-72', '73+']

    drivers['age'] = np.select(conds, choices)
    sorted_df = drivers.age.value_counts().sort_index().iloc[1:]
    sorted_df.plot.bar(color='tab:purple')
    plt.title('Number of Crashes by Age Group')
    plt.xlabel('Age Group')


def crashes_by_pedpedal_action(people):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    import pandas as pd
    f, ax = plt.subplots(figsize=(12, 8))
    sns.set_color_codes("muted")
    sns.barplot(x=people.pedpedal_action.value_counts().sort_values().values,
                y=people.pedpedal_action.value_counts().sort_values().index,
                data=people.pedpedal_action.value_counts().sort_values(),
                label="Injury Accidents", color="m", orient='h')
    pedpedal_sorted = people.pedpedal_action.value_counts().sort_values().index
    ax.set_yticklabels(pedpedal_sorted)
    ax.set_xlabel('Number of Crashes')
    ax.set_ylabel('Bicyclist Action')
    title = 'Number of Crashes based on Bicyclist Action (before Crash)'
    ax.set_title(title, size=15)
    sns.despine(left=True, bottom=True)


def crashes_by_damage(crashes):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    import pandas as pd
    crashes.damage.value_counts().plot.bar(color='tab:purple')
    plt.title('Number of Crashes based on Crash Estimated Damage')
    plt.xlabel('Damage (Estimated by Reporting Officer)')


def crashes_by_lighting_condition(crashes):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import pandas as pd
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    vc = crashes.lighting_condition.value_counts()
    vc.sort_values(ascending=False).plot.bar(color='tab:purple')
    plt.title('Number of Crashes based on Lighting Condition')


def airbag_deployment(people):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")

    people_ab = people[(people['airbag_deployed'] != 'DEPLOYMENT UNKNOWN') &
                       (people['airbag_deployed'] != 'NOT APPLICABLE')]

    conds = [(people_ab['airbag_deployed'] == 'DID NOT DEPLOY'),
             (people_ab['airbag_deployed'] != 'DID NOT DEPLOY')]
    choices = ['Did not deploy', 'Successfully Deployed']

    people_ab['airbag_deployed'] = np.select(conds, choices)

    no_inj_conds = (people_ab['injury_classification'] == 'NO INDICATION OF INJURY') | \
                   (people_ab['injury_classification'] == 'REPORTED, NOT EVIDENT')
    inj_conds = (people_ab['injury_classification'] != 'NO INDICATION OF INJURY') & \
                (people_ab['injury_classification'] != 'REPORTED, NOT EVIDENT')
    crash_no_inj = people_ab[no_inj_conds]
    crash_inj = people_ab[inj_conds]

    crash_no_inj = crash_no_inj.groupby('airbag_deployed').count()
    crash_inj = crash_inj.groupby('airbag_deployed').count()

    cni = crash_no_inj['crash_record_id']
    cyi = crash_inj['crash_record_id']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x=cni.index, y='crash_record_id', data=cni,
                label="No Injury Accidents", color="m")

    sns.set_color_codes("muted")
    sns.barplot(x=cyi.index, y='crash_record_id', data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="upper right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Airbag Deployed?")
    ax.set_title('Number of Crashes by Airbag Deployment', size=20)
    sns.despine(left=True, bottom=True)


def crashes_by_sex(people):

    crash_no_inj = people[(people['injury_classification'] == 'NO INDICATION OF INJURY') | (people['injury_classification'] == 'REPORTED, NOT EVIDENT')]
    crash_inj = people[(people['injury_classification'] != 'NO INDICATION OF INJURY') & (people['injury_classification'] != 'REPORTED, NOT EVIDENT')]

    crash_no_inj = crash_no_inj.groupby('sex').count()
    crash_inj = crash_inj.groupby('sex').count()

    cni = crash_no_inj['crash_record_id']
    cyi = crash_inj['crash_record_id']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x=cni.index, y='crash_record_id', data=cni,
                label="No Injury Accidents", color="m")

    sns.set_color_codes("muted")
    sns.barplot(x=cyi.index, y='crash_record_id', data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="upper right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Sex")
    ax.set_title('Number of Crashes by Sex', size=20)
    ax.set_xticklabels(['Female', 'Male', 'Other'])
    sns.despine(left=True, bottom=True)


def top_10_crash_sites(crashes):

    fig, ax = plt.subplots(figsize=(10, 6))

    street_names = crashes.street_name.value_counts(sort=True).index[:10]
    crash_frequency = crashes.street_name.value_counts(sort=True).values[:10]

    ax.barh(street_names, crash_frequency, color='tab:purple')
    ax.set_title('Top 10 Crash Sites (Street Names), Chicago IL', size=15)
    ax.set_yticklabels(street_names)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Crashes')
    ax.set_ylabel('Street Name')


def driver_vision(people):

    conds = [(people['driver_vision'] == 'NOT OBSCURED'),
             ((people['driver_vision'] != 'NOT OBSCURED') & (people['driver_vision'] != 'UNKNOWN'))]
    choices = ['N', 'Y']
    people['vision_obscured'] = np.select(conds, choices)

    crash_no_inj = people[(people['injury_classification'] == 'NO INDICATION OF INJURY') | (people['injury_classification'] == 'REPORTED, NOT EVIDENT')]
    crash_inj = people[(people['injury_classification'] != 'NO INDICATION OF INJURY') & (people['injury_classification'] != 'REPORTED, NOT EVIDENT')]

    crash_no_inj = crash_no_inj.groupby('vision_obscured').count()
    crash_inj = crash_inj.groupby('vision_obscured').count()

    cni = crash_no_inj['crash_record_id']
    cyi = crash_inj['crash_record_id']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(x=cni.index, y='crash_record_id', data=cni,
                label="No Injury Accidents", color="m")

    sns.barplot(x=cyi.index, y='crash_record_id', data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="upper right", frameon=True)
    ax.set(ylabel="Crashes", xlabel="Driver Vision Obscured?")
    ax.set_title('Number of Crashes by Driver Vision', size=20)
    ax.set_xticklabels(['Unknown', 'No', 'Yes'])
    sns.despine(left=True, bottom=True)


def driver_action(people):

    people_da = people[(people['driver_action'] != 'OTHER') & (people['driver_action'] != 'NONE') & (people['driver_action'] != 'UNKNOWN')]

    crash_no_inj = people_da[(people_da['injury_classification'] == 'NO INDICATION OF INJURY') | (people_da['injury_classification'] == 'REPORTED, NOT EVIDENT')]
    crash_inj = people_da[(people_da['injury_classification'] != 'NO INDICATION OF INJURY') & (people_da['injury_classification'] != 'REPORTED, NOT EVIDENT')]

    crash_no_inj = crash_no_inj.groupby('driver_action').count()
    crash_inj = crash_inj.groupby('driver_action').count()

    cni = crash_no_inj['crash_record_id']
    cyi = crash_inj['crash_record_id']

    cni = pd.DataFrame(cni).sort_values('crash_record_id', ascending=False).iloc[:8]
    cyi = pd.DataFrame(cyi).sort_values('crash_record_id', ascending=False).iloc[:8]

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x='crash_record_id', y=cni.index, data=cni,
                label="No Injury Accidents", color="m", orient='h')

    sns.set_color_codes("muted")
    sns.barplot(x='crash_record_id', y=cyi.index, data=cyi,
                label="Injury Accidents", color="r", orient='h')

    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="Driver Action", xlabel="Number of Crashes")
    ax.set_title('Number of Crashes sorted by Driver Action', size=20)
    sns.despine(left=True, bottom=True)


def seatbelt_used(people):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn')
    sns.set_theme(style="whitegrid")
    sns.set_color_codes("muted")
    import pandas as pd
    conds = [(people['safety_equipment'] == 'SAFETY BELT USED'),
             (people['safety_equipment'] == 'SAFETY BELT NOT USED')]
    choices = ['Yes', 'No']
    people['safety_belt_used'] = np.select(conds, choices)

    crash_no_inj = people[(people['injury_classification'] == 'NO INDICATION OF INJURY') | (people['injury_classification'] == 'REPORTED, NOT EVIDENT')]
    crash_inj = people[(people['injury_classification'] != 'NO INDICATION OF INJURY') & (people['injury_classification'] != 'REPORTED, NOT EVIDENT')]

    crash_no_inj = crash_no_inj.groupby('safety_belt_used').count()
    crash_inj = crash_inj.groupby('safety_belt_used').count()

    cni = crash_no_inj['crash_record_id']
    cyi = crash_inj['crash_record_id']

    cni = pd.DataFrame(cni)
    cyi = pd.DataFrame(cyi)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.barplot(x=cni.index, y='crash_record_id', data=cni,
                label="No Injury Accidents", color="m")

    sns.barplot(x=cyi.index, y='crash_record_id', data=cyi,
                label="Injury Accidents", color="r")

    ax.legend(ncol=2, loc="upper left", frameon=True)
    ax.set(ylabel="Number of Crashes", xlabel="Safety Belt Used?")
    ax.set_title('Number of Crashes by Seatbelt Usage', size=20)
    ax.set_xticklabels(['Unknown', 'No', 'Yes'])
    sns.despine(left=True, bottom=True)


def vehicle_defect(vehicles):

    vehicles.vehicle_defect.value_counts().sort_values(ascending=False)[3:].plot.bar(color='tab:purple')
    plt.title('Distribution of Vehicle Defects (if present)', size=15)
    plt.ylabel('Number of Crashes')
    plt.xlabel('Type of Vehicle Defect')


def class_imbalance(people):

    injuries = people.groupby('injury_classification').count()

    injuries = injuries.sort_values('crash_record_id', ascending=False)

    f, ax = plt.subplots(figsize=(10, 10))

    sns.set_color_codes("muted")
    sns.barplot(x='crash_record_id', y=injuries.index, data=injuries, color="m", orient='h')

    ax.set(ylabel="Type of Injury", xlabel="Number of Crashes (Millions)")
    ax.set_title('Number of Crashes by Injury Type', size=20)

    sns.despine(left=True, bottom=True)
