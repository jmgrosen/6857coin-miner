import urllib2
import json

def find_top_block(userName):
	pageLines = urllib2.urlopen("http://6857coin.csail.mit.edu/explore").read().splitlines()
	highest_height = 1E8
	highest_block_id = None

	for line in pageLines:
		if line.startswith("{id:"):
			#line contains block info
			block_id = line[5:69]
			#parse out height
			comma_index = line[77:].find(",")
			height = int(line[77:(77+comma_index)])
			#parse out name
			label_index = line.find("label")
			comma_index = line[label_index:].find("'")
			comma_index2 = line[label_index+comma_index+1:].find("'")
			name = line[label_index+comma_index+1:label_index+comma_index+comma_index2+1]

			if name == userName[:5]:
				if height < highest_height:
					highest_height = height 
					highest_block_id = block_id

	return highest_block_id


