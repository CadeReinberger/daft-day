from ranking_model import rerank, _rerank

def read_rankings(filename):
    #makes some pretty strong assumptions about correct use, so be careful
    rankings = []
    four_rankings = False
    with open(filename) as file:
        for line in file.readlines():
            try:
                start = line.index('(') + 1
                end = line.index(')')
                line = line[start: end]
                l = [a.strip() for a in line.split(',')]
                if len(l) == 4:
                    four_rankings = True
                    l[-1] = float(l[-1])
                rankings.append(tuple(l))
            except:
                pass
    return rankings, four_rankings

def get_reranked(filename):
    rankings, includes_adp = read_rankings(filename)
    return _rerank(rankings) if includes_adp else rerank(rankings)

def main():
    try:
        filename = input('Input rankings filename: ')
        res = get_reranked(filename)
        with open('output.txt', 'w') as file:
            file.writelines([str(l) + '\n' for l in res])
        print('Succesfully completed. Results in output.txt')
    except:
        print("Error Occured. Check rankings format. Otherwise it's a bug. Oops")
    
if __name__ == '__main__':
    main() 
    
        