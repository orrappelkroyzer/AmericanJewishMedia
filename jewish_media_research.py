import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import date, datetime
import ast
from scipy import stats
import json
import os
from sklearn.decomposition import PCA

def read_df(filename):
    df = pd.read_csv(filename)
    df = df.loc[df.score < 3]
    df['fixed'] = 1
    df['month'] = df.month.apply(date.fromisoformat)
    df['date'] = df.date.apply(lambda x: datetime.fromisoformat(x).replace(tzinfo=None))
    df['qaurter'] = df['date'].apply(lambda x: date(x.year, int((x.month-1)/3)*3+1, 1))
    df['tags'] = df.tags.fillna("[]").apply(parse_tags)
    df['section'] = df.section.fillna("[]").apply(lambda x: "[\"%s\"]" % x if x != "[]" else x).apply(parse_tags)
    df['title_score'] = df['google_sent_title_score']*df['google_sent_title_magnitude']
    df['desc_score'] = df['google_sent_desc_score']*df['google_sent_desc_magnitude']    
    df['title_score_normalized_magnitude'] = df['google_sent_title_score']*df['google_sent_title_normalized_magnitude'].fillna(0)
    df['desc_score_normalized_magnitude'] = df['google_sent_desc_score']*df['google_sent_desc_normalized_magnitude'].fillna(0)    
    
    for x in df.columns:
        if (not x.startswith("is") or not (x.endswith("tag") or x.endswith("section"))): continue
        df[x] = df[x].fillna(False)

    return df

    
def calc_score(df):
    pca = PCA(n_components=1)
    df['score'] = pca.fit_transform(np.array([df['title_score'], df['desc_score']]).T)
    if pca.components_[0][0] < 0:
        df['score'] *= (-1)
    pca1 = PCA(n_components=1)
    df['score_normalized_magnitude'] = pca1.fit_transform(np.array([df['title_score_normalized_magnitude'], df['desc_score_normalized_magnitude']]).T)
    if pca1.components_[0][0] < 0:
        df['score_normalized_magnitude'] *= (-1)
    return pca.explained_variance_ratio_[0], pca1.explained_variance_ratio_[0]

    
def plot(df, title, xlabel, ylabel, secondary_y=None, secondary_ylabel=None):
    plt.close()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if secondary_y is None:
        ax.plot(df)
        ax.legend(df.columns, loc='upper left')
    else:
        cols = df.columns.tolist()
        cols.remove(secondary_y)
        ax.plot(df[cols])
        ax.legend(df.columns, loc='upper left')
        ax2 = ax.twinx()
        colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
        color = colors[(len(cols)+1) % len(colors)]
        ax2.plot(df[secondary_y], color=color)
        ax2.legend([secondary_y], loc='upper right')
        ax2.set_ylabel(secondary_ylabel, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

def scatter(x, ys, title, xlabel, ylabel, with_formula=None, use_index=False):
    plt.close()
    fig, ax = plt.subplots()
    for y in ys.values():
        if use_index:
            ax.scatter(y.index, y.values, marker=".")
        else:   
            ax.scatter(x, y, marker=".")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(ys) > 1:
        ax.legend([k for k in ys.keys()], loc='upper right')
    if with_formula:
        colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']

        if use_index:
            min_x = min([min(y.index) for y in ys.values()])
            max_x = max([max(y.index) for y in ys.values()])
        else:
            min_x = min(x)
            max_x = max(x)
        min_x_for_plot = min_x
        max_x_for_plot = max_x
        linregs = {}
        i = 0
        for k, y in ys.items():
            if use_index:
                x_t = [t for t in y.index]
                if type(x_t[0]) in [date, pd._libs.tslibs.timestamps.Timestamp]:
                    x_t = [(t-min_x).days for t in x_t]
                    min_x_for_plot = 0
                    max_x_for_plot = (max_x - min_x).days

            else:
                x_t = x
            linregs[k] = linreg(x_t, y)
            ax.plot([min_x, max_x], [linregs[k][0]*min_x_for_plot+linregs[k][1], linregs[k][0]*max_x_for_plot+linregs[k][1]], 
                    color = colors[(len(ys)+1+i) % len(colors)])
            i += 1

        print(min_x + ((max_x-min_x)/100.))
        ax.text(min_x + ((max_x-min_x)/100.),  
                 max([y.max() for y in ys.values()]) - len(ys)*((max([y.max() for y in ys.values()])-min([y.min() for y in ys.values()]))/50.), 
                 "\n".join([linreg_text(linregs[k], xlabel, k) for k, y in ys.items()]))
    fig.tight_layout()
    plt.show()

def heatmap(x, y, title, xlabel, ylabel,  bins=10, range=None, cmax=None):
    plt.close()
    plt.hist2d(x, y, cmap='magma_r', bins=bins)
#    plt.hist2d(x, y, cmap='Oranges', bins=bins, range=range, cmax=cmax)
    cb = plt.colorbar()
    cb.set_label('# articles')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def barh(value, error, color, x_label, title):
    plt.close()
    y_pos = np.arange(len(value.index))
    fig, ax = plt.subplots()
    ax.barh(y_pos, value.values, xerr=error.values, color=color, align='center')
    ax.set_xlabel(x_label)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(value.index, rotation=30)
    ax.invert_yaxis()
    ax.set_title(title)
    plt.show()

def bar(df, error, y_label, title, width = 0.15):
    plt.close()
    x_pos = np.arange(len(df.columns))
    fig, ax = plt.subplots()
    index = 0
    for i, row in df.iterrows():
        print((index-len(df)/2)*width)
        ax.bar(x_pos+(index-len(df)/2)*width, row.values, width=width, yerr=error.loc[i].values, align='center')
        index += 1
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df.columns)
    ax.set_title(title)
    ax.legend(df.index)
    plt.show()


def count_over_time(df, title, axis, time_label='Month', mode="regular", additional_field_series=None):
    time_col = time_label.lower()
    if additional_field_series is None:
        additional_field_series = pd.Series(True, index=df.index)
    locs = {x :  df['is %s' % x] & additional_field_series for x in axis}
    t = {k : df.loc[v].groupby(time_col)['fixed'].count() for k, v in locs.items()}
    overall = df.groupby(time_col)['fixed'].count()
    secondary_y = None
    secondary_ylabel = None
    if mode == 'regular':
        t['Overall'] = overall
        t = pd.DataFrame(t)
        ylabel="# articles"
        sumcount = {k : int(v.sum()) for k, v in locs.items()}
    elif mode == 'percent':
        ylabel="%"
        if not additional_field_series.all():
            overall = {x : df.loc[df['is %s' % x]].groupby(time_col)['fixed'].count() for x in axis}
            t = pd.DataFrame({k: (v*100)/overall[k] for k, v in t.items()})
            t['Overall'] = df.loc[additional_field_series].groupby(time_col)['fixed'].count()*100/df.groupby(time_col)['fixed'].count()
            sumcount = additional_field_series.sum()/len(df)
        else:
            t = pd.DataFrame({k: v*100/overall for k, v in t.items()})
            t['Overall'] = overall
            secondary_y = "Overall"
            secondary_ylabel="# articles"
            sumcount = {k : v.sum()*100/additional_field_series.sum() for k, v in locs.items()}
    else:
        raise AssertionError("Received illegal mode %s" % mode)
    t = t.fillna(0)
    plot(t, title=title, xlabel=time_label, ylabel=ylabel, secondary_y=secondary_y)
    print(sumcount)
    return sumcount

def linreg(x, y):
    return stats.linregress(x, y)

def linreg_text(lr, x_label='x', y_label='y'):
    a = lr
    return "{} = {:4.2f}*{} + {:4.2f}  (R^2={:4.2f}, p={:4.2f})".format(y_label.lower().replace(" ", "_"), a[0], x_label.lower().replace(" ", "_"), a[1], a[2], a[3])

def parse_tags(x):
    try:
        if x[0] not in ["[", '"', '"'] :
            x = "[" + ", ".join(['"%s"' % y for y in  x.split(",")]) + "]"
        return ast.literal_eval(x)
    except:
        print(x)
        raise

def feature_hist(df, n_overall, col):
    histogram = {}
    def update_hist(x, histogram):
        if x not in histogram:
            histogram[x] = 0
        histogram[x] += 1
    df[col].apply(lambda x: [update_hist(y, histogram) for y in x] if type(x) == list else update_hist(x, histogram))
    histogram =  pd.DataFrame.from_dict({'count': histogram})
    histogram['percent'] = histogram['count']/len(df)
    histogram = histogram.sort_values(by='count', ascending=False)
    histogram = pd.DataFrame({'count' : [n_overall, n_overall-len(df)], 
                              'percent' : [1, (n_overall-len(df))/n_overall]}, 
                              index=['overall', 'untagged']).append(histogram)
    return histogram

def class_hist(df, class_list, n_overall, col):
    histogram = pd.DataFrame({'count' : [df["is %s" % x].sum() for x in class_list],
                              'percent' : [df["is %s" % x].sum()/len(df) for x in class_list]},
                              index = class_list)
    histogram = histogram.sort_values(by='count', ascending=False)
    ungrouped = (df[["is %s" % x for x in class_list]].sum(axis=1) == 0).sum()
    histogram = pd.DataFrame({'count' : [n_overall, n_overall-len(df), ungrouped], 
                              'percent' : [1, (n_overall-len(df))/n_overall, ungrouped/len(df)]},
                              index=['overall', 'untagged', 'ungrouped']).append(histogram)
    return histogram

def score_by_tag(df, tag_classes, title, score_field='score'):
    dfs = {k: df.loc[df["is %s tag" %k].fillna(False)] for k in tag_classes}
    colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
    t = pd.DataFrame(
        {'mean_score' : {k: v[score_field].mean() for k, v in dfs.items()},
         'sterr' : {k: v[score_field].std()/np.sqrt(len(v[score_field])) for k, v in dfs.items()},
         'color' : colors[0] })
    for i, tag in enumerate(['Israel related', 'Non-Israel related']):
        t.loc[tag, 'color'] = colors[i+1]
    t = t.sort_values(by='mean_score', ascending=False)
    barh(t.mean_score, t.sterr, t.color, "Mean score", "Mean Sentiment Score per Tag (%s)" % title)

def score_israel_vs_non_israel(df, axis, axis_name, title, score_field="score"):
    locs = {x: df["is %s" % x] for x in axis}
    t = {x : 
            {k: df.loc[v & df["is %s tag" %k].fillna(False)][score_field].mean() 
            for k in ["Israel related", "Non-Israel related"]} 
        for x, v in locs.items()}
    
    t_err = {x : 
                {k: df.loc[v & df["is %s tag" %k].fillna(False)][score_field].std() / \
                    np.sqrt(len( df.loc[v & df["is %s tag" %k].fillna(False)][score_field])) 
                for k in ["Israel related", "Non-Israel related"]} 
            for x, v in locs.items()}
    for x, v in locs.items():
        t[x]["Overall"] = df.loc[v][score_field].mean()
        t_err[x]["Overall"] = None
        
    bar(pd.DataFrame(t), error=pd.DataFrame(t_err), y_label="Mean score", title="Sentiment Score for Israel vs. Non Israel by %s (%s)" % (axis_name, title))
    

def read_data():
    d = r"C:\Users\orr\Dropbox\Documents\studies\CSUS\Jewish Media on Israel\V2.1"
    general_stats = {}
    publications = ['Forward', 'Tablet Magazine', 'Commentary', 'Jewish Journal']
    tag_classes = ['Israel related', 'American Jewry related', 'US News', 'World News', 'Religion', 'Holocaust', 'Anti-Semitism', 'Identity politics', 'Arts and Culture', 'Life', 'Other', 'Academia', 'Environment', 'US history']
    section_classes = ['Blogs', 'Culture', 'Life', 'News', 'Opinion', 'Other']
    df_all = read_df(os.path.join(d, "overall_meta_v1.csv"))
    general_stats['Count all'] = int(len(df_all))
    df_till_2009 = df_all.loc[(df_all.date > datetime(2009,4,1)) & (df_all.date < datetime(2019,6,1))]
    general_stats['Count till 2009'] = int(len(df_till_2009))
    tagged_loc = (df_till_2009['tags'].apply(len) > 0) | \
        (df_till_2009['is Commentary'] & \
            (df_till_2009[["is %s tag" % x for x in tag_classes]].sum(axis=1) > 0))
    sectioned_loc = (df_till_2009['section'].apply(len) > 0) | \
        (df_till_2009['is Commentary'] & \
            (df_till_2009[["is %s section" % x for x in section_classes]].sum(axis=1) > 0))
    df = df_till_2009.loc[tagged_loc & sectioned_loc]
    general_stats['Count tagged'] = int(len(df))
    general_stats['PCA explained variance'], general_stats['PCA explained variance normalized magnitude'] = calc_score(df)
    df_wo_top_percentile_score = df.loc[(df.google_sent_title_magnitude <= df.google_sent_title_magnitude.quantile(0.99)) &
                                             (df.google_sent_desc_magnitude <= df.google_sent_desc_magnitude.quantile(0.99))]
    general_stats['Count tagged w/o top percentile score'] = int(len(df_wo_top_percentile_score))
    general_stats['PCA explained variance w/o top percentile'], general_stats['PCA explained variance normalized magnitude w/o top percentile'] = calc_score(df_wo_top_percentile_score)
    df_wo_zero_score = df.loc[(abs(df.title_score) > 0.00001) | (abs(df.desc_score)> 0.00001)]
    general_stats['Count tagged w/o zero score'] = int(len(df_wo_zero_score))
    general_stats['PCA explained variance w/o zero'], general_stats['PCA explained variance normalized magnitude w/o zero'] = calc_score(df_wo_zero_score)
    df_wo_top_percentile_and_zero_score = df.loc[(df.google_sent_title_magnitude <= df.google_sent_title_magnitude.quantile(0.99)) &
                                             (df.google_sent_desc_magnitude <= df.google_sent_desc_magnitude.quantile(0.99)) &
                                             ((abs(df.title_score) > 0.00001) | (abs(df.desc_score)> 0.00001))]
    general_stats['Count tagged w/o top percentile and zero score'] = int(len(df_wo_top_percentile_and_zero_score))
    general_stats['PCA explained variance w/o top percentile and zero'], general_stats['PCA explained variance normalized magnitude w/o top percentile and zero score'] = calc_score(df_wo_top_percentile_and_zero_score)
    df_wo_top_percentile_and_zero_score_normalized_magnitude = df.loc[
        (df.google_sent_title_normalized_magnitude <= df.google_sent_title_normalized_magnitude.quantile(0.99)) &
        (df.google_sent_desc_normalized_magnitude <= df.google_sent_desc_normalized_magnitude.quantile(0.99)) &
        ((abs(df.title_score_normalized_magnitude) > 0.00001) | (abs(df.desc_score_normalized_magnitude)> 0.00001))]
    general_stats['Count tagged w/o zero score normalized magnitude']  = int(len(df_wo_top_percentile_and_zero_score_normalized_magnitude)) 
    general_stats['PCA explained variance w/o zero score normalized magnitude'], general_stats['PCA explained variance normalized magnitude w/o zero score and top percentile normalized magnitude'] = calc_score(df_wo_top_percentile_and_zero_score_normalized_magnitude)
    return d, publications, tag_classes, section_classes, df_all, df_till_2009, tagged_loc, sectioned_loc, df, df_wo_top_percentile_score,df_wo_zero_score, df_wo_top_percentile_and_zero_score, df_wo_top_percentile_and_zero_score_normalized_magnitude, general_stats

def main():

    d, publications, tag_classes, section_classes, df_all, df_till_2009, tagged_loc, sectioned_loc, df, df_wo_top_percentile_score,df_wo_zero_score, df_wo_top_percentile_and_zero_score, df_wo_zero_score_normalized_magnitude, general_stats = read_data()
    # dfs_by_score_cleaning = {"All Tagged Data " : df,
    #                     "Without Top Precentile" : df_wo_top_percentile_score,
    #                     "Without Zeros" : df_wo_zero_score, 
    #                     "Without Top Percentile and Zeros" : df_wo_top_percentile_and_zero_score}
    # dfs_by_score_nm_cleaning = {"All Tagged Data, Normalized Magnitude " : df,
    #                     "Without Zeros According to Normalized Magnitude" : df_wo_zero_score_normalized_magnitude}

    # # Articles count
    # general_stats['Publication hist all'] = count_over_time(df_all, axis=publications, title="Number of Articles per Month by Publication (all data)")
    # general_stats['Publication hist till 2009'] = count_over_time(df_till_2009, axis=publications, title="Number of Articles per Month by Publication (from 2009)")
    # general_stats['Publication hist tagged'] = count_over_time(df, axis=publications, title="Number of Tagged Articles per Month by Publication (from 2009)")
    # general_stats['Publication hist wo top percentile'] = count_over_time(df_wo_top_percentile_score, axis=publications, title="Number of Tagged Articles per Month w/o Top Percentile Score, by Publication (from 2009)")
    # general_stats['Publication hist wo zeros'] = count_over_time(df_wo_zero_score, axis=publications, title="Number of Tagged Articles per Month w/o Zero Score, by Publication (from 2009)")
    # general_stats['Publication hist wo zeros and top percentile'] = count_over_time(df_wo_top_percentile_and_zero_score, axis=publications, title="Number of Tagged Articles per Month w/o Top Percentile and Zero Score, by Publication (from 2009)")
    

    # # score graphs
    # for title, t_df in dfs_by_score_cleaning.items():
    #     heatmap(t_df['google_sent_title_score'], t_df['google_sent_title_magnitude'], title="Title Score Magnitude vs. Polarity (%s)" % title, xlabel="Polarity", ylabel="Magnitude",  bins=100, range=[[-1, 1], [0,10]])
    #     heatmap(t_df['google_sent_desc_score'], t_df['google_sent_desc_magnitude'], title="Description Score Magnitude vs. Polarity (%s)" % title, xlabel="Polarity", ylabel="Magnitude",  bins=100, range=[[-1, 1], [0,10]])
    #     heatmap(t_df['title_score'], t_df['desc_score'], title="Description Score vs. Title Score (%s)" % title, xlabel="Title score", ylabel="Descriptions score", bins=100, range=[[-1, 1], [-1,1]], cmax=10000)
    #     scatter(t_df['score'], {'Title score' : t_df['title_score'], 'Description score': t_df['desc_score']}, title="Description Score and Title Score vs. Combined Score (%s)" % title, xlabel="Combined score", ylabel="Score", with_formula=True)
    #     locs = t_df['desc_score'] == 0
    #     scatter(t_df.loc[locs, 'score'], {'Title score' : t_df.loc[locs, 'title_score']}, title="Title Score vs. Combined Score when Description Score is 0 (%s)" % title, xlabel="Combined score", ylabel="Title score", with_formula=True)
    # for title, t_df in dfs_by_score_nm_cleaning.items():
    #     heatmap(t_df['google_sent_title_score'], t_df['google_sent_title_normalized_magnitude'].fillna(0), title="Title Score Normalized Magnitude vs. Polarity (%s)" % title, xlabel="Polarity", ylabel="Magnitude",  bins=100, range=[[-1, 1], [0,10]])
    #     heatmap(t_df['google_sent_desc_score'], t_df['google_sent_desc_normalized_magnitude'].fillna(0), title="Description Score Normalized Magnitude vs. Polarity (%s)" % title, xlabel="Polarity", ylabel="Magnitude",  bins=100, range=[[-1, 1], [0,10]])
    #     heatmap(t_df['title_score_normalized_magnitude'], t_df['desc_score_normalized_magnitude'], title="Description Score vs. Title Score (%s)" % title, xlabel="Title score", ylabel="Descriptions score", bins=100, range=[[-1, 1], [-1,1]], cmax=10000)
    #     scatter(t_df['score_normalized_magnitude'], {'Title score' : t_df['title_score_normalized_magnitude'], 'Description score': t_df['desc_score_normalized_magnitude']}, title="Description Score and Title Score vs. Combined Score (%s)" % title, xlabel="Combined score", ylabel="Score", with_formula=True)
    #     locs = t_df['title_score_normalized_magnitude'] == 0
    #     scatter(t_df.loc[locs, 'score_normalized_magnitude'], {'Description score' : t_df.loc[locs, 'desc_score_normalized_magnitude']}, title="Description Score vs. Combined Score when Title Score is 0 (%s)" % title, xlabel="Combined score", ylabel="Description score", with_formula=True)



    # # tags histogram

    # for publication in publications:
    #     col = 'is %s' % publication
    #     pub_df = df.loc[df[col]]
        
    #     feature_hist(pub_df, len(df_till_2009[sectioned_loc & df_till_2009[col]]), 'tags').to_csv(os.path.join(d, "output", "tag_histogram", "%s.csv" % publication))
    #     class_hist(pub_df, ["%s tag" % y for y in tag_classes], len(df_till_2009[sectioned_loc & df_till_2009[col]]), 'tags').to_csv(os.path.join(d, "output", "tag_class_histogram", "%s.csv" % publication))
    #     feature_hist(pub_df, len(df_till_2009[tagged_loc & df_till_2009[col]]), 'section').to_csv(os.path.join(d, "output", "section_histogram", "%s.csv" % publication))
    #     class_hist(pub_df, ["%s section" % y for y in section_classes], len(df_till_2009[tagged_loc & df_till_2009[col]]), "section").to_csv(os.path.join(d, "output", "section_class_histogram", "%s.csv" % publication))

    # # Israel count
    # general_stats["Tag groups hist"] = count_over_time(df, axis=["%s tag" % x for x in tag_classes[:5]], 
    #                                                     title="Percent of Articles For Each Tag Group per Month", 
    #                                                     mode="percent")
    # for publication in publications:
    #     general_stats["Tag groups hist in %s" % publication] = \
    #         count_over_time(df.loc[df['is %s' % publication]], axis=["%s tag" % x for x in tag_classes[:5]], 
    #                                 title="Percent of Articles For Each Tag Group per Month in %s" % publication, 
    #                                 mode="percent")
    # for section in section_classes:
    #     general_stats["Tag groups hist in %s" % section] = \
    #         count_over_time(df.loc[df['is %s section' % section]], axis=["%s tag" % x for x in tag_classes[:5]], 
    #                                 title="Percent of Articles For Each Tag Group per Month in %s" % section, 
    #                                 mode="percent")
    # general_stats["israel in publications hist"] = count_over_time(df, axis=publications, 
    #                                                     title="Percent of Israel-Related Articles per Month by Publication", 
    #                                                     mode="percent",
    #                                                     additional_field_series = df['is Israel related tag'])
    # general_stats["israel in sections hist"] = count_over_time(df, axis=["%s section" % x for x in section_classes[:5]], 
    #                                                     title="Percent of Israel-Related Articles per Month by Section", 
    #                                                     mode="percent",
    #                                                     additional_field_series = df['is Israel related tag'])

    # # Israel vs. news
    # scatter(df.loc[df['is News section']].groupby('month')['fixed'].count()*100/df.groupby('month')['fixed'].count(), 
    #         {'% Israel' : df.loc[df['is Israel related tag']].groupby('month')['fixed'].count()*100/df.groupby('month')['fixed'].count()},
    #         title="% Israel per Month vs. % News per Month", 
    #         xlabel="% News", ylabel="% Israel", with_formula=True)

    # # score
    # for title, t_df in dfs_by_score_cleaning.items():
    #     score_by_tag(t_df, tag_classes + ["Non-Israel related"], title)
    #     score_israel_vs_non_israel(t_df, axis=publications, axis_name="Publication", title=title)
    #     score_israel_vs_non_israel(t_df, axis=["%s section" % x for x in section_classes[:5]], axis_name="Section", title=title)
    # for title, t_df in dfs_by_score_nm_cleaning.items():
    #     score_by_tag(t_df, tag_classes + ["Non-Israel related"], title, score_field="score_normalized_magnitude")
    #     score_israel_vs_non_israel(t_df, axis=publications, axis_name="Publication", title=title, score_field="score_normalized_magnitude")
    #     score_israel_vs_non_israel(t_df, axis=["%s section" % x for x in section_classes[:5]], axis_name="Section", title=title, score_field="score_normalized_magnitude")


    # score over time
    scatter(None,
            {x : df_wo_top_percentile_and_zero_score.loc[df_wo_top_percentile_and_zero_score['is Israel related tag'] & df_wo_top_percentile_and_zero_score['is %s' % x]].groupby('date')['score'].mean() for x in publications},
            title="Israel Daily Mean Score over Time by Publication", 
            xlabel="Date", ylabel="Mean score", with_formula=True,
            use_index=True)
    scatter(None,
            {x : df_wo_top_percentile_and_zero_score.loc[df_wo_top_percentile_and_zero_score['is Israel related tag'] & df_wo_top_percentile_and_zero_score['is %s section' % x]].groupby('date')['score'].mean() for x in section_classes[:5]},
            title="Israel Daily Mean Score over Time by Section", 
            xlabel="Date", ylabel="Mean score", with_formula=True,
            use_index=True)
    scatter(None,
            {x : df_wo_zero_score_normalized_magnitude.loc[df_wo_zero_score_normalized_magnitude['is Israel related tag'] & df_wo_zero_score_normalized_magnitude['is %s' % x]].groupby('date')['score_normalized_magnitude'].mean() for x in publications},
            title="Israel Daily Mean Score over Time by Publication (According to Normalized Magnitude)", 
            xlabel="Date", ylabel="Mean score", with_formula=True,
            use_index=True)
    scatter(None,
            {x : df_wo_zero_score_normalized_magnitude.loc[df_wo_zero_score_normalized_magnitude['is Israel related tag'] & df_wo_zero_score_normalized_magnitude['is %s section' % x]].groupby('date')['score_normalized_magnitude'].mean() for x in section_classes[:5]},
            title="Israel Daily Mean Score over Time by Section (According to Normalized Magnitude)", 
            xlabel="Date", ylabel="Mean score", with_formula=True,
            use_index=True)



    # with open(os.path.join(d, "general_stats_1.json"), 'w') as f:
    #     json.dump(general_stats, f, indent=4)
    # print(json.dumps(general_stats, indent=4))

if __name__ == "__main__":
    main()