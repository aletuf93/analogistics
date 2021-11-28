
import geocoder
import time

mykey = ''  # insert here your bing API key


def bruteforceGeocoding(dict_address: dict, apiKey: str, waitTime: int):
    """
    Geocoder trying the Bing API given the address

    Args:
        dict_address (dict): dictionary with the address to locate.
        apiKey (str): Bing API key.
        waitTime (int): Time to wait between a call and the following.

    Returns:
        a (TYPE): json results from the geocoder.

    """

    time.sleep(waitTime)
    if ('ADDRESS' in dict_address.keys()) & ('CITY' in dict_address.keys()) & ('ZIPCODE' in dict_address.keys()):
        g = geocoder.bing(location=None, addressLine=dict_address['ADDRESS'],
                          locality=dict_address['CITY'], postalCode=dict_address['ZIPCODE'],
                          method='details', key=apiKey)
        a = g.json
        if a is not None:
            return a

    time.sleep(waitTime)
    if ('ADDRESS' in dict_address.keys()) & ('CITY' in dict_address.keys()):
        g = geocoder.bing(location=None, addressLine=dict_address['ADDRESS'], locality=dict_address['CITY'], method='details', key=apiKey)
        a = g.json
        if a is not None:
            return a

    time.sleep(waitTime)
    if ('CITY' in dict_address.keys()):
        g = geocoder.bing(location=None, locality=dict_address['CITY'],
                          method='details', key=apiKey)
        a = g.json
        if a is not None:
            return a

    # if arrived here, there is nothing found
    return None


def addressGeocoder(dict_address: dict, apiKey: str = mykey, waitTime: int = 1):
    """
    Geocoder trying the Bing API given an address

    Args:
        dict_address (dict): dictionary with the address to locate.
        apiKey (str, optional): Bing API key. Defaults to mykey.
        waitTime (int, optional): Time to wait between a call and the following. Defaults to 1.

    Returns:
        dict: Output dictionary with geocoding information.

    """

    # query bing
    a = bruteforceGeocoding(dict_address, apiKey, waitTime)

    result = {}
    if a is None:
        print("**GEOCODER: No geodata found using BING")
        return []
    else:

        if 'lat' in a.keys():
            result['LATITUDE_api'] = a['lat']

        if 'lng' in a.keys():
            result['LONGITUDE_api'] = a['lng']

        if 'address' in a.keys():
            result['ADDRESS_api'] = a['address']

        if 'city' in a.keys():
            result['CITY_api'] = a['city']

        if 'country' in a.keys():
            result['COUNTRY_api'] = a['country']

        if 'state' in a.keys():
            result['STATE_api'] = a['state']

        if 'postal' in a.keys():
            result['ZIPCODE_api'] = a['postal']

        print(f"**GEOCODER: geodata found at {result}")
        return result


def directGeocoder(dict_geo: dict, apiKey: str = mykey, waitTime: int = 1) -> dict:
    """
    use a dictionary with latitude and longitude to find information on the geopoint

    Args:
        dict_geo (dict): Input dictionary with latitude and longitude.
        apiKey (str, optional): bing API Keys. Defaults to mykey.
        waitTime (int, optional): Time to wait between calls. Defaults to 1.

    Returns:
        dict: Output dictionary with address.

    """

    result = {'LATITUDE_api': dict_geo['LATITUDE'],
              'LONGITUDE_api': dict_geo['LONGITUDE'],
              }

    time.sleep(waitTime)
    if ('LATITUDE' in dict_geo.keys()) & ('LONGITUDE' in dict_geo.keys()):
        g = geocoder.bing([dict_geo['LATITUDE'], dict_geo['LONGITUDE']], method='reverse', key=apiKey)
        a = g.json
    if a is None:
        print("**GEOCODER: No geodata found using BING")
        return []
    else:

        if 'lat' in a.keys():
            result['LATITUDE_api'] = a['lat']

        if 'lng' in a.keys():
            result['LONGITUDE_api'] = a['lng']

        if 'address' in a.keys():
            result['ADDRESS_api'] = a['address']

        if 'city' in a.keys():
            result['CITY_api'] = a['city']

        if 'country' in a.keys():
            result['COUNTRY_api'] = a['country']

        if 'state' in a.keys():
            result['STATE_api'] = a['state']

        if 'postal' in a.keys():
            result['ZIPCODE_api'] = a['postal']

        print(f"**GEOCODER: geodata found at {result}")
        return result
