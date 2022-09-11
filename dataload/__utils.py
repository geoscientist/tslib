from datetime import date, timedelta

def get_date_range(start, end):
    sdate = date(int(start.split('-')[0]), int(start.split('-')[1]), int(start.split('-')[2]))   # start date
    edate = date(int(end.split('-')[0]), int(end.split('-')[1]), int(end.split('-')[2]))   # end date

    delta = edate - sdate

    drange = []       

    for i in range(delta.days + 1):
        day = sdate + timedelta(days=i)
        drange.append(day)

    return drange
