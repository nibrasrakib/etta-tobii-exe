import requests
from bs4 import BeautifulSoup
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36'
}
urls = []
usernames = []
for i in range(1, 21):
    url = 'https://embed.expertfile.com/v1/organization/5338/1?page_number=' + \
        str(i) + '&hide_search_category=on&profile_url=custom&url_override=https%3A%2F%2Funcnews.unc.edu%2Fexperts%2Fprofile%3Fexpert%3D%7B%7Busername%7D%7D&tag=all&access=public&page_size=10&font_family=Open+Sans%2C+Helvetica+Neue%2C+Helvetica%2C+Arial%2C+sans-serif&other_font_name=&other_font_source=&url_color=%232376A1&color=%23333333&background_color=%23ffffff&hide_search_bar=no&hide_search_sort=yes&hide_search_category=no&ajax_request=yes'
    r = requests.get(url)
    data = r.json()

    experts = data['data']['experts']
    for expert in experts:
        username = expert['user']['username']
        usernames.append(username)
        urls.append(
            'https://uncnews.unc.edu/experts/profile?expert=' + username)

dict_expert = {}

for username in usernames:
    url = "https://embed.expertfile.com/v1/expert/" + username + "/1?profile_url=expertfile&amp;profile_url=expertfile&amp;expert=jonathan.abramowitzphd&amp;font_family=Open+Sans%2C+Helvetica+Neue%2C+Helvetica%2C+Arial%2C+sans-serif&amp;other_font_name=&amp;other_font_source=&amp;url_color=%232376A1&amp;color=%23333333&amp;background_color=%23ffffff&amp;url_override="
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    name = soup.find('h1')
    biography = ''
    industry_expertise = []
    title = ''
    title_des = ''
    
    # biography
    if soup.find(id='biography') is not None:
        biography_sec = soup.find(id='biography')
        if biography_sec.find(class_='content') is not None:
            biography = biography_sec.find(class_='content').getText()
    
    # industry expertise
    if soup.find('ul', class_='cols') is not None:
        for expertise in soup.find('ul', class_='cols').find_all('li'):
            industry_expertise.append(expertise.text)
    areas_expertise = []
    for expertise in soup.find_all('em', class_='topic-tag'):
        areas_expertise.append(expertise.text)
    
    # education
    # I rewrote code in this section as the original code might result in mismatch. The original code is commented out.
    all_educations = []
    education_dict = {}
    if soup.find(id='education') is not None:
        all_educations = soup.find(id='education').find_all(class_='display-item')
        for education_ in all_educations:
            if education_.find('h4') is not None:
                school = education_.find('h4').getText()
                degree = ''
                if education_.find('p') is not None:
                    degree = education_.find('p').getText()
                if school in education_dict:
                    education_dict[school] += '; '
                    education_dict[school] += degree
                else:
                    education_dict[education_.find('h4').getText()] = degree
    '''
    schools = []
    if soup.find(id='education') is not None:
        schools = soup.find(id='education').find_all('h4')
    degrees = []
    if soup.find(id='education') is not None:
        degrees = soup.find(id='education').find_all('p',{"class":None})
    education_dict = {}
    for i in range(len(schools)):
        education_dict[schools[i].text] = degrees[i].text
    '''
    
    # affiliation
    affiliations = []
    if soup.find(id='affiliations') is not None:
        for affiliation in soup.find(id='affiliations').find('ul').find_all('li'):
            affiliations.append(affiliation.text)
    
    # title and title description
    if soup.find(class_="col-xs-9") is not None:
        if soup.find(class_="col-xs-9").find('strong') is not None:
            title = soup.find(class_="col-xs-9").find('strong').getText()
        des_li = soup.find(class_="col-xs-9").find_all('p',{'class': None})
        if des_li != []:
            for p in des_li:
                if p.find('strong') == None:
                    title_des = p.getText()           
            
    temp_dict = {}
    temp_dict['url'] = 'https://uncnews.unc.edu/experts/profile?expert=' + username
    if len(biography) > 0:
        temp_dict['biography'] = biography
    if len(title) > 0:
        temp_dict['title'] = title
    if len(title_des) > 0:
        temp_dict['title_description'] = title_des
    if len(industry_expertise) > 0:
        temp_dict['industry_expertise'] = industry_expertise
    if len(areas_expertise) > 0:
        temp_dict['areas_expertise'] = areas_expertise
    if len(education_dict) > 0:
        temp_dict['education'] = education_dict
    if len(affiliations) > 0:
        temp_dict['affiliations'] = affiliations
    
    dict_expert[name.text] = temp_dict
print((dict_expert))

f_out = open('E:/PATTIE_Rep/files/expert_database.txt','w', encoding='utf-8') 
f_out.write(json.dumps(dict_expert, ensure_ascii=True))
