import xml.etree.ElementTree as ET
print('start parsing...')
tree = ET.parse('/Users/xinzhaoli/Documents/Research/pubmed21n0005.xml')
root = tree.getroot()
# types = root.iter('PublicationType')
# pubTypes = set()
# for pubType in types:
#     pubTypes.add(pubType.text)
# print(pubTypes)
for article in root:
    source = article.iter('JournalIssue')
    if not source:
        print('Found article without JournalIssue: ')
        print(list(article.Iter('ArticleTitle')[0].text))
print('end parsing...')
